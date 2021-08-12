import sys, os

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
    for k, v in d[dataset].keys():
        original_csv_path = os.path.join(dataset, mapping_folder, k+".csv")
        with open(original_csv_path, 'r') as f:
            ls = f.readlines()
        print(k, len(ls), v)

run(sys.argv[1])