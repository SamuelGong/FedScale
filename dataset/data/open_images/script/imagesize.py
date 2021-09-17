import os
from PIL import Image

dataset='validation'
files = [os.path.join(dataset, x) for x in os.listdir(dataset)]

for file in files:
    im = Image.open(file)
    width, height = im.size
    print(width, height)
