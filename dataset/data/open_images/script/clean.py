import pickle, os
import csv

client_mapping_f = open('imageToAuthor', 'rb')
client_mapping = pickle.load(client_mapping_f)

tag_idx = {}
with open('classTags') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        tag_idx[line.strip()] = len(tag_idx)

# input, output, client_id
path = './train'
img_files = os.listdir(path)

results = []

for file in img_files:
    items = file.split('__')
    img_name = items[0]
    img_tag = items[1].split('.')[0]

    results.append([file, tag_idx[img_tag], client_mapping[img_name]])

results.sort(key=lambda k:(k[2],k[1]))
results = [['img_name', 'img_tag', 'client_id']] + results

csvFile = open('train_manifest.csv', "w")
writer = csv.writer(csvFile)

for line in results:
    writer.writerow(line)
csvFile.close()
