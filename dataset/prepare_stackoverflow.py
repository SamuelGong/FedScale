import os
import gc
import shutil
import collections
import copy
import torch
import numpy as np
import pickle
import csv
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader, Dataset
import zipfile

N_JOBS = 16

# dependencies
#   torch
#   transformer
#   sentencepiece

# debugging
import time

# configurations
repack_train = False
repack_test = True
test_training = False
prepare_num_training_clients = 1000
prepare_num_testing_clients = 10

model_name = "albert-base-v2"
root_dir = "data/reddit"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")

gen_dir = os.path.join(root_dir, "Reddit")
train_gen_dir = os.path.join(gen_dir, 'train')
test_gen_dir = os.path.join(gen_dir, 'test')


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


def prepare_data(data_dir, block_size, num_files_clip):
    files = [entry.name for entry
             in os.scandir(data_dir) if '_cached_lm_' not in entry.name]
    # make sure files are ordered
    files = [os.path.join(data_dir, x) for x in sorted(files)]
    files = files[:num_files_clip]

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
        if true_sample_clients:
            user_id_base = true_sample_clients[-1] + 1

    print(f'\tNumber of samples processed: {len(inputs)}.')
    return inputs, labels, client_mapping, sample_clients


def repack_data(raw_clients, gen_dir, starting_cnt=1):
    client_cnt = starting_cnt
    client_samples_cnts = []
    for raw_client_id, sample_id_list in raw_clients.items():
        client_path = os.path.join(gen_dir, str(client_cnt))
        os.makedirs(client_path, exist_ok=True)
        file_path = os.path.join(client_path, 'data.bin')

        inputs = []
        labels = []
        for sample_id in sample_id_list:
            inputs.append(train_inputs[sample_id])
            labels.append(train_labels[sample_id])
        client_samples_cnts.append(len(sample_id_list))

        data_dict = {
            'x': inputs,
            'y': labels
        }

        with open(file_path, 'wb') as fout:
            pickle.dump(data_dict, fout)

        zipfile_path = os.path.join(gen_dir, str(client_cnt) + '.zip')
        with zipfile.ZipFile(zipfile_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(client_path)
        shutil.rmtree(client_path)

        client_cnt += 1
    print(f"\t# clients: {len(client_samples_cnts)}.\n\t"
          f"min/max/avg # samples: {min(client_samples_cnts)}"
          f"/{max(client_samples_cnts)}"
          f"/{np.mean(client_samples_cnts)}.")
        
        
def read_data_map(mapping_path, num_clients):
    sample_id = 0
    with open(mapping_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        read_first = True
        raw_train_clients = {}

        for row in csv_reader:
            if read_first:
                read_first = False
            else:
                client_id = row[0]

                if client_id not in raw_train_clients:
                    if len(raw_train_clients.keys()) \
                            == num_clients:
                        break
                    raw_train_clients[client_id] = []

                raw_train_clients[client_id].append(sample_id)
                sample_id += 1
    return sample_id, raw_train_clients


# File operations
start_time = time.perf_counter()
os.makedirs(gen_dir, exist_ok=True)
if repack_train:
    if os.path.isdir(train_gen_dir):
        shutil.rmtree(train_gen_dir)
if repack_test:
    if os.path.isdir(test_gen_dir):
        shutil.rmtree(test_gen_dir)
os.makedirs(train_gen_dir, exist_ok=True)
os.makedirs(test_gen_dir, exist_ok=True)

# Importing libraries
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
block_size = 64 - (tokenizer.model_max_length
                   - tokenizer.max_len_single_sentence)
print(f"[Debug] NLP libraries imported. Computed block size: {block_size}. "
      f"Elapsed time: {time.perf_counter() - start_time}")

# Reading Mapping information for training datasets
train_data_clip, raw_train_clients = read_data_map(
    train_mapping_path, prepare_num_training_clients)
print(f"Training data mapping read. "
      f"Elapsed time: {time.perf_counter() - start_time}")

test_data_clip, _ = read_data_map(
    test_mapping_path, prepare_num_testing_clients
)
print(f"Testing data mapping read. "
      f"Elapsed time: {time.perf_counter() - start_time}")

# Reading and packing training data
if repack_train or test_training:
    train_inputs, train_labels, train_client_mapping, train_sample_clients \
            = prepare_data(train_data_dir, block_size, num_files_clip=train_data_clip)
    print(f"Training data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    if repack_train:
        repack_data(raw_train_clients, train_gen_dir, starting_cnt=1)
        print(f"Training data packed. "
              f"Elapsed time: {time.perf_counter() - start_time}")

    if not test_training:
        del train_inputs
        del train_labels
        del train_client_mapping
        del train_sample_clients
        gc.collect()

# Reading and packing testing data
if repack_test or test_training:
    test_inputs, test_labels, test_client_mapping, test_sample_clients \
            = prepare_data(test_data_dir, block_size, num_files_clip=test_data_clip)
    print(f"Testing data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    if repack_test:
        raw_test_clients = {
            'mock_client': [sample_id for sample_id in range(len(test_inputs))]
        }
        repack_data(raw_test_clients, test_gen_dir, starting_cnt=0)
        print(f"Testing data packed. "
              f"Elapsed time: {time.perf_counter() - start_time}")

    if not test_training:
        del test_inputs
        del test_labels
        del test_client_mapping
        del test_sample_clients
        gc.collect()

# Testing training with loaded data
if test_training:
    train_batch_size = 20
    train_dataset = TextDataset(train_inputs, train_labels)
    train_data = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                            shuffle=True, drop_last=True)
    print(f"Testing training. Number of training data batches {len(train_data)}. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    test_batch_size = 20
    test_dataset = TextDataset(test_inputs, test_labels)
    test_data = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                            shuffle=True, drop_last=False)
    print(f"Number of testing data batches {len(test_data)}. "
          f"Elapsed time: {time.perf_counter() - start_time}")

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
                test_loss_value = 0.0
                for test_inputs, test_targets in test_data:
                    test_outputs = model(test_inputs, labels=test_targets)
                    test_loss = test_outputs[0]
                    test_loss_value += test_loss.data.item()

                test_loss_value /= len(test_data)
                perplexity = np.exp(test_loss_value)
                model.train()

                test_time = time.perf_counter() - start_time
                print(f"[Evaluate] "
                      f"perplexity: {perplexity}, elapsed time: {test_time}.")

            if completed_steps == local_steps:
                break