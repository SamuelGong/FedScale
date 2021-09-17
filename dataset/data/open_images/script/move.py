#!/usr/bin/env python3

import os
from os import listdir
from os.path import isfile, join

prefix =  ['b']

for pre in prefix:
    path = './crop' + str(pre) 
    imgFiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and '.jpg' in f]

    print("Working on {}".format(path))
    for item in imgFiles:
        os.system('mv ' + item + ' ./train/')

