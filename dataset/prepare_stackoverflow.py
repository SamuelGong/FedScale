import os
import gc
import collections
import copy
import numpy as np
from multiprocessing import Pool, cpu_count

N_JOBS = 16
N_JOBS = 1

# dependencies
# torch
# transformer
# sentencepiece

# debugging
import time

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
model = AutoModelForMaskedLM.from_config(config)
tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=True)

# relative path to this file
root_dir = "data/reddit"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    inputs = np.array(inputs)
    labels = copy.deepcopy(inputs)

    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    mlm_probability = 0.15
    probability_matrix = np.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels
    ]
    probability_matrix = np.ma.array(probability_matrix, mask=special_tokens_mask)
    probability_matrix = probability_matrix.filled(fill_value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = ma.masked_equal(labels, tokenizer.pad_token_id).mask.astype(int)
        probability_matrix = np.ma.array(probability_matrix, mask=padding_mask)
        probability_matrix = probability_matrix.filled(fill_value=0.0)

    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype(bool) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(bool) & masked_indices & ~indices_replaced
    random_words = np.random.randint(0, len(tokenizer), labels.shape, dtype=np.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.tolist(), labels.tolist()


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


print(f"[Debug] [A] Elapsed time: {time.perf_counter() - start_time}")


def feature_creation_worker(files, tokenizer, block_size, worker_idx):
    inputs = []
    labels = []
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
                examples = tokenizer\
                    .build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                print(f'examples: {examples}')
                input, label = mask_tokens(examples, tokenizer)
                inputs += input
                labels += labels
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

    return (inputs, labels, client_mapping, sample_client)

block_size = 64 - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
print(f"[Debug] block_size: {block_size}")

inputs = []
labels = []
sample_client = []
client_mapping = collections.defaultdict(list)
user_id = -1

files = [entry.name for entry in os.scandir(train_data_dir) if '_cached_lm_' not in entry.name]
# make sure files are ordered
files = [os.path.join(train_data_dir, x) for x in sorted(files)]

files = files[:10]

print(f"[Debug] [B] Elapsed time: {time.perf_counter() - start_time}")

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
for (input, label, client_mapping, sample_client) in pool_outputs:
    inputs += input
    labels += label

    true_sample_client = [i + user_id_base for i in sample_client]
    sample_client += true_sample_client
    for user_id, true_user_id in zip(sample_client, true_sample_client):
        client_mapping[true_user_id] = client_mapping[user_id]
    user_id_base = true_sample_client[-1] + 1

print(f"[Debug] [C] Elapsed time: {time.perf_counter() - start_time}")