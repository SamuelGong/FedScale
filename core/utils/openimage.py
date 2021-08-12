from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import csv

class OpenImage():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, dataset='train', transform=None, target_transform=None, imgview=False,
                 filter_less=None, filter_more=None):
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_file = dataset # 'train', 'test', 'val'

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You have to download it')

        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data, self.targets = self.load_file(self.path, filter_less, filter_more)
        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.path, imgName))
        
        # avoid channel error
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path, filter_less=None, filter_more=None):
        datas, labels = [], []
        client_data_dict = {}

        if filter_less or filter_more:
            valid_data_set = set()
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        if row[0] not in client_data_dict:
                            client_data_dict[row[0]] = [row[1]]
                        else:
                            client_data_dict[row[0]].append(row[1])
                    line_count += 1

            for _, client_data in client_data_dict.items():
                if filter_less and filter_more and filter_less <= len(client_data) <= filter_more:
                    _ = [valid_data_set.add(i) for i in client_data]
                elif filter_less and filter_less <= len(client_data):
                    _ = [valid_data_set.add(i) for i in client_data]
                elif filter_more and len(client_data) <= filter_more:
                    _ = [valid_data_set.add(i) for i in client_data]

            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        if row[1] not in valid_data_set:
                            continue
                        datas.append(row[1])
                        labels.append(int(row[-1]))
                    line_count += 1
        else:
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        datas.append(row[1])
                        labels.append(int(row[-1]))
                    line_count += 1

        return datas, labels

    def load_file(self, path, filter_less=None, filter_more=None):

        # load meta file to get labels
        datas, labels = self.load_meta_data(os.path.join(
            self.processed_folder, 'client_data_mapping', self.data_file+'.csv'), filter_less, filter_more)

        return datas, labels
