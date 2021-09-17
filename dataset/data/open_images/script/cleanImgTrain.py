import os

files = os.listdir("/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/train")

with open('imgTrainMapping', 'w') as fout:
	for entry in files:
		fout.writelines(str(entry) + '\n')
