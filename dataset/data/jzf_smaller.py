import sys, os
import numpy as np
import shutil

mapping_folder = "client_data_mapping"
postfix = "_smaller"
d = {
    "google_speech": {
        "train": 10800,
        "test": 1800
    },
    "open_images": {
        "train": 120000,
        "val": 20000
    },
    "reddit" : {
        "train": 600000,
        "test": 100000
    }
}

def run(dataset):
    np.random.seed(seed=233)
    if dataset == "reddit":
        original_data_path = os.path.join(dataset, k)
        dirs = os.listdir(original_data_path)
        print(len(dirs))
    else:
        for k, v in d[dataset].items():
            original_csv_path = os.path.join(dataset, mapping_folder,
                                             k + ".csv")
            smaller_csv_path = os.path.join(dataset, mapping_folder,
                                            k + postfix + ".csv")

            with open(original_csv_path, 'r') as f:
                ls = f.readlines()

            new_ls = [ls[0]] # the first line is header
            choice = sorted(np.random.choice(len(ls) - 1, v, replace=False))
            for i in choice:
                new_ls.append(ls[i + 1])

            with open(smaller_csv_path, 'w') as f:
                f.writelines(new_ls)

            original_data_path = os.path.join(dataset, k)
            smaller_data_path = os.path.join(dataset, k + postfix)

            if os.path.exists(smaller_data_path):
                shutil.rmtree(smaller_data_path)
            os.makedirs(smaller_data_path)

            per_cent = max(1, v // 100)
            for idx, l in enumerate(new_ls[1:]):
                if idx % per_cent == 0:
                    print(f"{k}: {idx}/{v}")

                file = l.split(",")[1]
                os.system(f"cp {os.path.join(original_data_path, file)} {smaller_data_path}")

run(sys.argv[1])