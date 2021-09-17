import os, csv
import gc, collections
import pickle

gc.disable()


def read_csv(path, request_col):
	ans = []
	with open(path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count != 0:
				ans.append([row[x] for x in request_col])
			line_count += 1
	return ans

ans = read_csv('train_raw.csv', [0, 1])

client_mapping = collections.defaultdict(list)

for client_id, sample_id in ans:
	client_mapping[client_id].append(int(sample_id))

mapping_keys = list(client_mapping.keys())
for client_id in mapping_keys:
	if len(client_mapping[client_id]) < 20 or len(client_mapping[client_id]) > 500:
		del client_mapping[client_id]

client_ids = list(client_mapping.keys())
client_ids.sort()

with open('../train/_cached_lm_62_raw', 'rb') as fin:
	data = pickle.load(fin)
	mapping = pickle.load(fin)

results = [['client_id', 'sample_path', 'label_name', 'label_id']]
new_data = []
for idx, client_id in enumerate(client_ids):
	for sample_id in client_mapping[client_id]:
		results.append([idx, len(results)-1, -1, -1])
		new_data.append(data[sample_id])

with open('../train/_cached_lm_62', 'wb') as fout:
	pickle.dump(new_data, fout)
	pickle.dump(mapping, fout)


csvFile = open('train.csv', "w")
writer = csv.writer(csvFile)

for line in results:
    writer.writerow(line)
csvFile.close()
