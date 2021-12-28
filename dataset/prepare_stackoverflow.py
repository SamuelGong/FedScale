import os
import gc
import collections
import copy
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader, Dataset

# N_JOBS = 16
N_JOBS = 2

# dependencies
#   torch
#   transformer
#   sentencepiece

# debugging
import time

# relative path to this file
root_dir = "data/reddit"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")

start_time = time.perf_counter()

# Albert over StackOverflow
model_name = "albert-base-v2"
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AlbertTokenizer,
    MobileBertForPreTraining,
    AutoModelForMaskedLM
)
config = AutoConfig.from_pretrained(model_name)
tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=True)

print(f"[Debug] NLP libraries imported. "
      f"Elapsed time: {time.perf_counter() - start_time}")


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original. """
    inputs = np.array(inputs)
    labels = copy.deepcopy(inputs)

    # We sample a few tokens in each sequence for masked-LM training
    # (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    mlm_probability = 0.15
    probability_matrix = np.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(
            val, already_has_special_tokens=True) for val in labels
    ]
    probability_matrix = np.ma.array(probability_matrix, mask=special_tokens_mask)
    probability_matrix = probability_matrix.filled(fill_value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = np.ma.masked_equal(labels, tokenizer.pad_token_id).mask.astype(int)
        probability_matrix = np.ma.array(probability_matrix, mask=padding_mask)
        probability_matrix = probability_matrix.filled(fill_value=0.0)

    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(
        1, np.full(labels.shape, 0.8)).astype(bool) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = np.random.binomial(
        1, np.full(labels.shape, 0.5)).astype(bool) & masked_indices & ~indices_replaced
    random_words = np.random.randint(0, len(tokenizer), labels.shape, dtype=np.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.tolist(), labels.tolist()


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.long)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        return inputs, labels


def feature_creation_worker(files, tokenizer, block_size, worker_idx):
    examples = []
    sample_client = []
    client_mapping = collections.defaultdict(list)

    user_id = -1
    start_time = time.time()
    for idx, file in enumerate(files):
        try:
            with open(file, encoding="utf-8", errors='ignore') as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            if len(tokenized_text) > 0:
                user_id += 1

            for i in range(0, len(tokenized_text) -
                              block_size + 1, block_size):  # Truncate in block of block_size
                example = tokenizer\
                    .build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                examples.append(example)
                client_mapping[user_id].append(len(examples)-1)
                sample_client.append(user_id)
        except Exception as e:
            print(f"CPU worker {worker_idx}: fail due to {e}")
            raise e

        if idx % 1000 == 0:
            print(f"CPU worker {worker_idx}: {len(files)-idx} "
                  f"files left, {idx} files complete, remaining "
                  f"time {(time.time()-start_time)/(idx+1)*(len(files)-idx)}")
            gc.collect()

    inputs, labels = mask_tokens(examples, tokenizer)
    return (inputs, labels, client_mapping, sample_client)


def prepare_data(data_dir, block_size, clip):
    files = [entry.name for entry
             in os.scandir(data_dir) if '_cached_lm_' not in entry.name]
    # make sure files are ordered
    files = [os.path.join(data_dir, x) for x in sorted(files)]
    files = files[:clip]

    inputs = []
    labels = []
    sample_clients = []
    client_mapping = collections.defaultdict(list)
    user_id = -1

    pool_inputs = []
    pool = Pool(N_JOBS)
    worker_cnt = 0
    for begin, end in chunks_idx(range(len(files)), N_JOBS):
        pool_inputs.append([files[begin:end], tokenizer, block_size, worker_cnt])
        worker_cnt += 1

    pool_outputs = pool.starmap(feature_creation_worker, pool_inputs)
    pool.close()
    pool.join()

    user_id_base = 0
    for (input, label, client_mapping, sample_clients) in pool_outputs:
        inputs += input
        labels += label

        true_sample_clients = [i + user_id_base for i in sample_clients]
        sample_clients += true_sample_clients
        for user_id, true_user_id in zip(sample_clients, true_sample_clients):
            client_mapping[true_user_id] = client_mapping[user_id]
        user_id_base = true_sample_clients[-1] + 1

    return inputs, labels, client_mapping, sample_clients


block_size = 64 - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

train_inputs, train_labels, train_client_mapping, train_sample_clients \
    = prepare_data(train_data_dir, block_size, clip=200)
train_batch_size = 20
train_dataset = TextDataset(train_inputs, train_labels)
train_data = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                        shuffle=True, drop_last=True)
print(f"[Debug] Training data loaded. "
      f"Elapsed time: {time.perf_counter() - start_time}")

test_inputs, test_labels, test_client_mapping, test_sample_clients \
    = prepare_data(test_data_dir, block_size, clip=20)
test_batch_size = 20
test_dataset = TextDataset(test_inputs, test_labels)
test_data = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                        shuffle=True, drop_last=False)
print(f"[Debug] Testing data loaded. "
      f"Elapsed time: {time.perf_counter() - start_time}")


### Testing training
model = AutoModelForMaskedLM.from_config(config)
learning_rate = 4e-5

weight_decay = 0.0

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
# len(train_data) = 269 when files = files[:200]

local_steps = 30
completed_steps = 0
test_interval = 5

model.train()
while completed_steps < local_steps:
    for inputs, targets in train_data:
        outputs = model(inputs, labels=targets)
        loss = outputs[0]
        loss_value = loss.data.item()

        train_time = time.perf_counter() - start_time
        print(f"[Step {completed_steps}] "
              f"loss: {loss_value}, elapsed time: {train_time}.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed_steps += 1

        if completed_steps % test_interval == 0:
            model.eval()
            loss_value = 0.0
            for test_inputs, test_targets in test_data:
                test_outputs = model(test_inputs, test_targets)
                loss = test_outputs[0]
                loss_value += loss.data.item()

            perplexity = np.exp(loss_value)
            model.train()

            test_time = time.perf_counter() - start_time
            print(f"[Evaluate] "
                  f"perplexity: {perplexity}, elapsed time: {test_time}.")

        if completed_steps == local_steps:
            break