import os
import pickle
from numpy import array
import numpy as np 

processed_folder = '/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg'
# load class information
with open(os.path.join(processed_folder, 'classTags'), 'r') as fin:
    classes = [tag.strip() for tag in fin.readlines()]

classMapping = {_class: i for i, _class in enumerate(classes)}

def load_file(path):
    rawImg, rawTags = [], []

    imgFiles = os.scandir(path)
    #imgFiles = [f for f in os.listdir(path)]# if os.path.isfile(os.path.join(path, f)) and '.jpg' in f]

    for imgFile in imgFiles:
        imgFile = imgFile.name
        classTag = imgFile.replace('.jpg', '').split('__')[1]
        if classTag in classMapping:
            rawImg.append(imgFile)
            rawTags.append(classMapping[classTag])

    return rawImg, rawTags

def partitionTrace(dataToClient, rawImg, rawTags):
    clientToData = {}
    clientNumSamples = {}
    numOfLabels = 596

    # data share the same index with labels
    for index, sample in enumerate(rawImg):
        sample = sample.split('__')[0]
        clientId = dataToClient[sample]
        labelId = rawTags[index]

        if clientId not in clientToData:
            clientToData[clientId] = []
            clientNumSamples[clientId] = [0] * numOfLabels

        clientToData[clientId].append(index)
        clientNumSamples[clientId][labelId] += 1

    clientClass = np.zeros([len(clientNumSamples), numOfLabels])

    for idx, client in enumerate(clientNumSamples.keys()):
        clientClass[idx] = clientNumSamples[client]

    print(clientClass[0])
    with open('openimg_data.pkl', 'wb') as fout:
        pickle.dump(clientClass, fout)
        
    # # normalize each client
    # for client in clientNumSamples:
    #     samplesClient = sum(clientNumSamples[client])
    #     clientNumSamples[client] = array(clientNumSamples[client])/float(samplesClient)

    # with open('clientSampleVec', 'wb') as fout:
    #     pickle.dump(clientNumSamples, fout)


with open(os.path.join(processed_folder, 'imageToAuthor'), 'rb') as db:
    dataToClient = pickle.load(db)

rawImg, rawTags = load_file(os.path.join(processed_folder, 'train'))
partitionTrace(dataToClient, rawImg, rawTags)
