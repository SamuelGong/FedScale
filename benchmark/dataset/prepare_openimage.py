import logging
import os
import gc
import time
import shutil
from PIL import Image
import numpy as np
import pickle
import csv
from multiprocessing import Pool, cpu_count
import zipfile
import tarfile

N_JOBS = cpu_count()


# def jpg_handler(files, worker_idx):
#     examples = []
#
#     st = time.time()
#     for idx, f in enumerate(files):
#         try:
#             image = Image.open(f)
#             # avoid channel error: some images are greyscale
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
#
#             example = np.asarray(image)
#             examples.append(example)
#         except Exception as e:
#             print(f"CPU worker {worker_idx}: fail due to {e}", flush=True)
#             raise e
#
#         if idx % 1000 == 0:
#             print(f"CPU worker {worker_idx}: {len(files)-idx} "
#                   f"files left, {idx} files complete, remaining "
#                   f"time {(time.time()-st)/(idx+1)*(len(files)-idx)}", flush=True)
#             gc.collect()
#
#     return examples

# configurations
repack_train = True
repack_test = True
# after repacking, can upload to s3 using commands like
#   aws s3 cp jzf_openImg s3://jiangzhifeng/openImage --recursive

prepare_num_training_clients = 1000
# e.g., ~18800s for rgcpu7

prepare_num_testing_clients = 50
# e.g., ~100s for rgcpu7

# feature_creation_worker = jpg_handler
root_dir = "data/openImg"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")

gen_dir = os.path.join(root_dir, "jzf_openImg")
train_gen_dir = os.path.join(gen_dir, 'train')
test_gen_dir = os.path.join(gen_dir, 'test')

start_time = time.perf_counter()
os.makedirs(gen_dir, exist_ok=True)
if repack_train:
    if os.path.isdir(train_gen_dir):
        raise ValueError(f'Please remove {train_gen_dir} manually')
        # shutil.rmtree(train_gen_dir)
if repack_test:
    if os.path.isdir(test_gen_dir):
        raise ValueError(f'Please remove {test_gen_dir} manually')
        # shutil.rmtree(test_gen_dir)
os.makedirs(train_gen_dir, exist_ok=False)
os.makedirs(test_gen_dir, exist_ok=False)


# Reading Mapping information for training datasets
def read_data_map(mapping_path, num_clients):
    read_first = True
    client_map = {}

    with open(mapping_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if read_first:
                read_first = False
            else:
                # client_id,sample_path,label_name,label_id
                client_id = row[0]
                sample_path = row[1]
                label = int(row[3])

                if client_id not in client_map:
                    if len(client_map.keys()) \
                            == num_clients:
                        break
                    client_map[client_id] = {
                        'sample_paths': [],
                        'labels': []
                    }

                client_map[client_id]['sample_paths'].append(sample_path)
                client_map[client_id]['labels'].append(label)
    return client_map


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def _repack_raw_data(client_map, begin, end, worker_idx, gen_dir, starting_cnt):
    st = time.time()
    client_cnt = begin + starting_cnt  # start from starting_cnt
    client_samples_cnts = []

    for idx, raw_client_id in enumerate(list(
            client_map.keys()
    )[begin:end]):
        client_dict = client_map[raw_client_id]
        sample_paths = client_dict["sample_paths"]
        labels = client_dict["labels"]

        sample_label_map = {k: v for k, v in zip(sample_paths, labels)}
        sample_label_map_file = os.path.join(gen_dir, f'{client_cnt}_sample_label_map')
        with open(sample_label_map_file, 'wb') as fout:
            pickle.dump(sample_label_map, fout)

        client_samples_cnts.append(len(labels))
        tar_path = os.path.join(gen_dir, f"{client_cnt}.tar")
        # use prefix "{client_cnt}/' for ease of extraction
        with tarfile.open(tar_path, "w:") as tar:
            tar.add(sample_label_map_file, arcname=f"{client_cnt}/sample_label_map")
            for sample_path in sample_paths:
                arcname = f"{client_cnt}/{data_file}"
                data_file = os.path.join(train_data_dir, sample_path)
                tar.add(data_file, arcname=arcname)
        os.remove(sample_label_map_file)

        client_cnt += 1
        if idx % 10 == 0:
            print(f"CPU worker {worker_idx}: {end-begin-idx-1} "
                  f"clients left, {idx + 1} clients' data packed, remaining "
                  f"time {(time.time()-st)/(idx+1)*(end-begin-idx-1)}", flush=True)
            gc.collect()

    return client_samples_cnts


# New
def repack_raw_data(client_map, gen_dir, starting_cnt=1):
    pool_inputs = []
    pool = Pool(N_JOBS)
    worker_cnt = 0
    # split_factor = 16  # to avoid too large return values for each subprocess
    for begin, end in chunks_idx(range(len(client_map)), N_JOBS):
        pool_inputs.append([client_map, begin, end, worker_cnt, gen_dir, starting_cnt])
        worker_cnt += 1

    pool_outputs = pool.starmap(_repack_raw_data, pool_inputs)
    pool.close()
    pool.join()

    all_client_samples_cnts = []
    for client_samples_cnts in pool_outputs:
        all_client_samples_cnts += client_samples_cnts

    print(f"\t# clients: {len(all_client_samples_cnts)}.\n\t"
          f"min/max/avg # samples: {min(all_client_samples_cnts)}"
          f"/{max(all_client_samples_cnts)}"
          f"/{np.mean(all_client_samples_cnts)}.")


if repack_train:
    train_client_map = read_data_map(
        mapping_path=train_mapping_path,
        num_clients=prepare_num_training_clients
    )
    print(f"Training data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    repack_raw_data(train_client_map, train_gen_dir, starting_cnt=1)
    print(f"Training data packed. "
          f"Elapsed time: {time.perf_counter() - start_time}")


if repack_test:
    test_client_map = read_data_map(
        mapping_path=test_mapping_path,
        num_clients=prepare_num_testing_clients
    )
    print(f"Testing data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    raw_test_clients = {
        'mock_client': [sample_id for sample_id in range(len(test_client_map))]
    }
    repack_raw_data(test_client_map, test_gen_dir, starting_cnt=0)
    print(f"Testing data packed. "
          f"Elapsed time: {time.perf_counter() - start_time}")

