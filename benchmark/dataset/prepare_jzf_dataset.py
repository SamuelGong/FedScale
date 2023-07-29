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

import time

# configurations
repack_train = True
repack_test = True
# after repacking, can upload to s3 using commands like
#   aws s3 cp Reddit s3://jiangzhifeng/Reddit --recursiv

prepare_num_training_clients = 1
# e.g., Reddit 1000: ~870s

prepare_num_testing_clients = 5
# e.g., Reddit 10: ~10s

#root_dir = "data/reddit"
root_dir = "data/openImg"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")

feature_creation_worker = jpg_handler

gen_dir = os.path.join(root_dir, "jzf_openImg")
train_gen_dir = os.path.join(gen_dir, 'train')
test_gen_dir = os.path.join(gen_dir, 'test')

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

# Reading Mapping information for training datasets
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
def jpg_hander(files, worker_idx):
    examples = []
    sample_client = []
    client_mapping = collections.defaultdict(list)

    user_id = -1
    start_time = time.time()
    for idx, f in enumerate(files):
        try:
            with open(f, encoding="utf-8", errors='ignore') as fin:
                text = fin.read()

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


def prepare_data(data_dir, num_files_clip):
    files = [entry.name for entry in os.scandir(data_dir)]
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
        pool_inputs.append([files[begin:end], worker_cnt])
        worker_cnt += 1

    pool_outputs = pool.starmap(feature_creation_worker, pool_inputs)
    pool.close()
    pool.join()

    user_id_base = 0
    for (input, label, m, s) in pool_outputs:
        inputs += input
        labels += label

        true_sample_clients = [i + user_id_base for i in s]
        sample_clients += true_sample_clients
        for user_id, true_user_id in zip(s, true_sample_clients):
            client_mapping[true_user_id] = m[user_id]
        if true_sample_clients:
            user_id_base = true_sample_clients[-1] + 1

    print(f'\tNumber of samples processed: {len(inputs)}.')
    return inputs, labels, client_mapping, sample_clients

def repack_data(raw_clients, inputs, labels, gen_dir, starting_cnt=1):
    client_cnt = starting_cnt
    client_samples_cnts = []
    for raw_client_id, sample_id_list in raw_clients.items():
        file_path = os.path.join(gen_dir, 'data.bin')

        client_inputs = []
        client_labels = []
        for sample_id in sample_id_list:
            client_inputs.append(inputs[sample_id])
            client_labels.append(labels[sample_id])
        client_samples_cnts.append(len(sample_id_list))

        data_dict = {
            'x': client_inputs,
            'y': client_labels
        }

        with open(file_path, 'wb') as fout:
            pickle.dump(data_dict, fout)

        zipfile_path = os.path.join(gen_dir, str(client_cnt) + '.zip')
        with zipfile.ZipFile(zipfile_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path, arcname=str(client_cnt))

        os.remove(file_path)
        client_cnt += 1

    print(f"\t# clients: {len(client_samples_cnts)}.\n\t"
          f"min/max/avg # samples: {min(client_samples_cnts)}"
          f"/{max(client_samples_cnts)}"
          f"/{np.mean(client_samples_cnts)}.")

if repack_train:
    train_inputs, train_labels, train_client_mapping, train_sample_clients \
            = prepare_data(train_data_dir, num_files_clip=train_data_clip)
    print(f"Training data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    repack_data(raw_train_clients, train_inputs, train_labels,
                train_gen_dir, starting_cnt=1)
    print(f"Training data packed. "
          f"Elapsed time: {time.perf_counter() - start_time}")


if repack_test:
    test_inputs, test_labels, test_client_mapping, test_sample_clients \
            = prepare_data(test_data_dir, num_files_clip=test_data_clip)
    print(f"Testing data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    raw_test_clients = {
        'mock_client': [sample_id for sample_id in range(len(test_inputs))]
    }
    repack_data(raw_test_clients, test_inputs, test_labels,
                test_gen_dir, starting_cnt=0)
    print(f"Testing data packed. "
          f"Elapsed time: {time.perf_counter() - start_time}")

