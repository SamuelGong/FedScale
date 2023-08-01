import csv

num_clients = 1000

mapping_path = 'data/openImg/client_data_mapping/train.csv'
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

empty_data_list = []
for client_id in range(1, num_clients + 1):
    if client_id not in raw_clients:
        empty_data_list.append(client_id)

print(client_id)
