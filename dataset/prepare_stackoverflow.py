import os

# relative path to this file
root_dir = "data/reddit"
client_data_mapping_path = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
test_data_dir = os.path.join(root_dir, "test")

# test
print(os.isfile(client_data_mapping_path))
print(os.isdir(train_data_dir))
print(os.isdir(test_data_dir))
