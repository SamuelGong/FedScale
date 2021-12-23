import os

# relative path to this file
root_dir = "data/reddit"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")

# test
print(os.path.isfile(train_mapping_path))
print(os.path.isfile(test_mapping_path))
print(os.path.isdir(train_data_dir))
print(os.path.isdir(test_data_dir))
