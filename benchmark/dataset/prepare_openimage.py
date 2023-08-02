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
        shutil.rmtree(train_gen_dir)
if repack_test:
    if os.path.isdir(test_gen_dir):
        shutil.rmtree(test_gen_dir)
os.makedirs(train_gen_dir, exist_ok=True)
os.makedirs(test_gen_dir, exist_ok=True)


# Reading Mapping information for training datasets
def read_data_map(mapping_path, num_clients):
    sample_id = 0
    labels = []
    with open(mapping_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        read_first = True
        raw_clients = {}

        for row in csv_reader:
            if read_first:
                read_first = False
            else:
                # client_id,sample_path,label_name,label_id
                client_id = row[0]
                label = int(row[3])

                if client_id not in raw_clients:
                    if len(raw_clients.keys()) \
                            == num_clients:
                        break
                    raw_clients[client_id] = []

                raw_clients[client_id].append(sample_id)
                labels.append(label)
                sample_id += 1
    return sample_id, raw_clients, labels


train_data_clip, raw_train_clients, train_labels = read_data_map(
    train_mapping_path, prepare_num_training_clients)
print(f"Training data mapping read. "
      f"Elapsed time: {time.perf_counter() - start_time}")

test_data_clip, _, test_labels = read_data_map(
    test_mapping_path, prepare_num_testing_clients
)
print(f"Testing data mapping read. "
      f"Elapsed time: {time.perf_counter() - start_time}")


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


# New: Locating data
def prepare_data_path(data_dir, num_files_clip):
    files = [entry.name for entry in os.scandir(data_dir)]
    # make sure files are ordered
    files = [os.path.join(data_dir, x) for x in sorted(files)]
    print(f'\tNumber of samples processed: {len(files)}.')
    return files[:num_files_clip]


# # Old: Reading and packing training data
# def prepare_data(data_dir, num_files_clip):
#     files = [entry.name for entry in os.scandir(data_dir)]
#     # make sure files are ordered
#     files = [os.path.join(data_dir, x) for x in sorted(files)]
#     files = files[:num_files_clip]
#
#     pool_inputs = []
#     pool = Pool(N_JOBS)
#     worker_cnt = 0
#     split_factor = 16  # to avoid too large return values for each subprocess
#     for begin, end in chunks_idx(range(len(files)), N_JOBS * split_factor):
#         pool_inputs.append([files[begin:end], worker_cnt])
#         worker_cnt += 1
#
#     pool_outputs = pool.starmap(feature_creation_worker, pool_inputs)
#     pool.close()
#     pool.join()
#
#     all_examples = []
#     for examples in pool_outputs:
#         all_examples += examples
#     print(f'\tNumber of samples processed: {len(all_examples)}.')
#     return all_examples

def _repack_raw_data(raw_clients, begin, end, worker_idx,
                     example_paths, labels, gen_dir, starting_cnt):
    st = time.time()
    client_cnt = begin + starting_cnt  # start from starting_cnt
    client_samples_cnts = []

    for idx, raw_client_id in enumerate(list(
            raw_clients.keys()
    )[begin:end]):
        sample_id_list = raw_clients[raw_client_id]
        label_path = os.path.join(gen_dir, f'{client_cnt}_label.bin')

        client_examples_path = []
        client_labels = []
        for sample_id in sample_id_list:
            client_examples_path.append(example_paths[sample_id])
            client_labels.append(labels[sample_id])
        client_samples_cnts.append(len(sample_id_list))

        # for client_example_file in client_examples_path:
        #     shutil.copy(client_example_file, d)
        # client_samples_cnts.append(len(client_examples_path))
        with open(label_path, 'wb') as fout:
            pickle.dump(client_labels, fout)

        # To be faster and avoid errors on extraction:
        # "tarfile.ReadError: unexpected end of data"

        tar_path = os.path.join(gen_dir, f"{client_cnt}.tar")
        with tarfile.open(tar_path, "w:") as tar:
            arcname = f"{client_cnt}/{label_path.split('/')[-1]}"
            tar.add(label_path, arcname=arcname)
            for client_example_file in client_examples_path:
                arcname = f"{client_cnt}/{client_example_file.split('/')[-1]}"
                tar.add(client_example_file, arcname=arcname)
        os.remove(label_path)

        client_cnt += 1
        if idx % 10 == 0:
            print(f"CPU worker {worker_idx}: {end-begin-idx-1} "
                  f"clients left, {idx + 1} clients' data packed, remaining "
                  f"time {(time.time()-st)/(idx+1)*(end-begin-idx-1)}", flush=True)
            gc.collect()

    return client_samples_cnts


# New
def repack_raw_data(raw_clients, example_paths, labels, gen_dir, starting_cnt=1):
    pool_inputs = []
    pool = Pool(N_JOBS)
    worker_cnt = 0
    # split_factor = 16  # to avoid too large return values for each subprocess
    for begin, end in chunks_idx(range(len(raw_clients)), N_JOBS):
        pool_inputs.append([raw_clients, begin, end, worker_cnt,
                            example_paths, labels, gen_dir, starting_cnt])
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


# # Old
# def repack_data(raw_clients, examples, labels, gen_dir, starting_cnt=1):
#     client_cnt = starting_cnt
#     client_samples_cnts = []
#     for raw_client_id, sample_id_list in raw_clients.items():
#         # though zipped files are even larger
#         temp_file_path = os.path.join(gen_dir, 'data.bin')
#         # temp_file_path = os.path.join(gen_dir, str(client_cnt))
#
#         client_inputs = []
#         client_labels = []
#         for sample_id in sample_id_list:
#             client_inputs.append(examples[sample_id])
#             client_labels.append(labels[sample_id])
#         client_samples_cnts.append(len(sample_id_list))
#
#         data_dict = {
#             'x': client_inputs,
#             'y': client_labels
#         }
#         with open(temp_file_path, 'wb') as fout:
#             pickle.dump(data_dict, fout)
#         zipfile_path = os.path.join(gen_dir, str(client_cnt) + '.zip')
#         with zipfile.ZipFile(zipfile_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
#             zf.write(temp_file_path, arcname=str(client_cnt))
#
#         os.remove(temp_file_path)
#         client_cnt += 1
#
#     print(f"\t# clients: {len(client_samples_cnts)}.\n\t"
#           f"min/max/avg # samples: {min(client_samples_cnts)}"
#           f"/{max(client_samples_cnts)}"
#           f"/{np.mean(client_samples_cnts)}.")


if repack_train:
    train_examples_path = prepare_data_path(train_data_dir, num_files_clip=train_data_clip)
    print(f"Training data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    repack_raw_data(raw_train_clients, train_examples_path, train_labels,
                train_gen_dir, starting_cnt=1)
    print(f"Training data packed. "
          f"Elapsed time: {time.perf_counter() - start_time}")


if repack_test:
    test_examples_path = prepare_data_path(test_data_dir, num_files_clip=test_data_clip)
    print(f"Testing data read. "
          f"Elapsed time: {time.perf_counter() - start_time}")

    raw_test_clients = {
        'mock_client': [sample_id for sample_id in range(len(test_examples_path))]
    }
    repack_raw_data(raw_test_clients, test_examples_path, test_labels,
                test_gen_dir, starting_cnt=0)
    print(f"Testing data packed. "
          f"Elapsed time: {time.perf_counter() - start_time}")

