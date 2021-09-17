import os, pickle, random, csv
import numpy as np
import shutil

train = os.scandir('./train')
test = os.scandir('./test')

client_list = []
for audio in train:
    audio = audio.name
    client = audio.split('_')[1]
    if client not in client_list:
    	client_list.append(client)
    
for audio in test:
    audio = audio.name
    client = audio.split('_')[1]
    if client not in client_list:
    	client_list.append(client)
print(len(client_list))
test_list = client_list[2187:2403]
val_list = client_list[2403:]
print(test_list)
print(val_list)

train = os.scandir('./train')
test = os.scandir('./test')

test_result = []
val_result = []

for audio in test:
    temp = audio.name.split("_")
    c = temp[1]
    if temp[1] in test_list:
    	test_result.append([client_list.index(temp[1]), audio.name, temp[0], "-1"])
    elif temp[1] in val_list:
    	val_result.append([client_list.index(temp[1]), audio.name, temp[0], "-1"])
    	shutil.move('test/'+audio.name, 'val/'+audio.name)
	
random.shuffle(test_result)
random.shuffle(val_result)


with open("test.csv", "w") as fout:
    writer = csv.writer(fout, delimiter=',')
    writer.writerow(['client_id' , 'data_path' , 'label_name', 'label_id'])
    for r in test_result:
    	writer.writerow(r)


with open("val.csv", "w") as fout:
    writer = csv.writer(fout, delimiter=',')
    writer.writerow(['client_id' , 'data_path' , 'label_name', 'label_id'])
    for r in val_result:
    	writer.writerow(r)














"""
client_dict = {k: v for v, k in enumerate(client_list)}
labels = ['up', 'two', 'sheila', 'zero', 'yes', 'five', 'one', 'happy', 'marvin', 'no', 'go', 'seven', 'eight', 'tree', 'stop', 'down', 'forward', 'learn', 'house', 'three', 'six', 'backward', 'dog', 'cat', 'wow', 'left', 'off', 'on', 'four', 'visual', 'nine', 'bird', 'right', 'follow', 'bed']

label_dict = {k: v for v, k in enumerate(labels)}

#with open('client_index', 'wb') as f:
#    pickle.dump(client_dict, f)


train = os.scandir('./train')
test = os.scandir('./test')

distr = {}
for audio in train:
    audio = audio.name
    client = audio.split('_')[1]
    label = audio.split('_')[0]
    if client_dict[client] not in distr:
        distr[client_dict[client]] = [0] * len(labels)
    distr[client_dict[client]][label_dict[label]] += 1

    
for audio in test:
    audio = audio.name
    client = audio.split('_')[1]
    label = audio.split('_')[0]
    if client_dict[client] not in distr:
        distr[client_dict[client]] = [0] * len(labels)
    distr[client_dict[client]][label_dict[label]] += 1

distr_list = [v for _, v in distr.items()]
print(len(distr_list))
with open('speech_data_distr.pkl', 'wb') as f:
    pickle.dump(distr_list, f)
"""
"""
print("Done with distr")
print(sampled[:10])

sampled_size = np.sum(sampled, axis = 1)
with open('speech_data_size_10k.pkl', 'wb') as f:
    pickle.dump(sampled_size, f)
"""
    
    



"""
train = os.scandir('./train')
audioToClient = {}
for audio in train:
    audio = audio.name
    client = audio.split('_')[1]
    print(client)
    print(audio)
    audioToClient[audio] = client_dict[client]

print(audioToClient)
with open('audioToClient', 'wb') as f:
    pickle.dump(audioToClient, f)
"""
