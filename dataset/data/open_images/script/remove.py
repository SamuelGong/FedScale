import os

path = "/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/test"
files = os.scandir(path)

for entry in files:
	fileName = os.path.join(path, entry.name)
	if os.path.getsize(fileName) < 3:
		os.remove(fileName)
