import csv
import random
from os import listdir
from os.path import isfile, join
import os
from collections import OrderedDict
import pickle
import collections

random.seed(233)

imgIdToAuthor = OrderedDict()
authorIds = {}
authorSet = {}
resultSet = {}

set_name = 'test'

# Load img csv
with open(f'{set_name}-images-with-rotation.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')

    for row in csv_reader:
        imgId = row[0].strip()#.replace('.jpg', '')
        author = row[5].strip()

        imgIdToAuthor[imgId] = author

fileNames = os.scandir(f'/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/{set_name}')
author_set = collections.defaultdict(int)

for file in fileNames:
    ImgName = file.name.split('__')[0]
    author = imgIdToAuthor[ImgName]

    if author not in authorSet:
        authorSet[author] = len(authorSet)
    
    resultSet[ImgName] = authorSet[author]
    author_set[authorSet[author]] += 1
bars = 20
count = 0
count_32 = 0
for key in author_set:
    if author_set[key] >= 16:
        count += 1
    if author_set[key] >= bars:
        count_32 +=1 

print(len(author_set), count, count_32)


tag_idx = {}
with open('classTags') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        tag_idx[line.strip()] = len(tag_idx)

# input, output, client_id
img_files = os.listdir(set_name)

results = []
#bars = 20

for file in img_files:
    items = file.split('__')
    img_name = items[0]
    img_tag = items[1].split('.')[0]

    if img_tag in tag_idx:
        if author_set[resultSet[img_name]] >= bars:
            results.append([resultSet[img_name], file, img_tag, tag_idx[img_tag]])
    else:
        print(f"can not find tag {img_tag}")
        os.system(f'rm {set_name}/{file}')

results.sort(key=lambda k:(k[0],k[-1]))
results = [['client_id', 'sample_path', 'label_name', 'label_id']] + results

csvFile = open(f'./client_data_mapping/{set_name}_{bars}.csv', "w")
writer = csv.writer(csvFile)

for line in results:
    writer.writerow(line)
csvFile.close()
print(count_32, len(results)-1)

# with open('imageToAuthor_url', 'wb') as fout:
#     pickle.dump(resultSet, fout)
