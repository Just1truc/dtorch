""" imports """
import requests
from typing import Union, Tuple
import os
from abc import ABC, abstractmethod
import dtorch as dt
import gzip
import numpy as np
import struct
from dtorch.dataset import JDataset

""" classes """

    
class MNISTDataset(JDataset):

    def __init__(self, root: str = './data', split: Tuple[str] | str = ('train', 'test'), download: bool = False) -> None:
        """The mnist dataset made by yan lecun

        Args:
            root (str, optional): folder of the datasets. Defaults to './data'.
            split (Tuple[str] | str, optional): split return format. Defaults to ('train', 'test').
            download (bool, optional): download dataset?. Defaults to False.
        """

        self.__dataset_path : str = [
            os.path.join(root, path)
            for path in [
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz'
            ]
        ]
        self.__dataset_urls : str = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ]

        self.__train_x : dt.jtensors.JTensors = None
        self.__train_y : dt.jtensors.JTensors = None
        self.__test_x : dt.jtensors.JTensors = None
        self.__test_y : dt.jtensors.JTensors = None

        super().__init__(root, split, download)


    @property
    def _dataset_exists(self) -> bool:
        return all([os.path.exists(path) for path in self.__dataset_path])
    

    """ methods """

    @property
    def data(self) -> Tuple[Tuple[dt.jtensors.JTensors]]:
        if (isinstance(self.split, Tuple)):
            train = (self.__train_x, self.__train_y)
            test = (self.__test_x, self.__test_y)
            return (train, test) if self.split[0] == 'train' else (test, train)
        return (self.__train_x, self.__train_y) if 'train' else (self.__test_x, self.__test_y)


    def download(self) -> None:
        print("downloading")

        assert (len(self.__dataset_path) == len(self.__dataset_urls))

        for i in range(len(self.__dataset_urls)):
            with open(self.__dataset_path[i], 'wb+') as bf:
                bf.write(requests.get(self.__dataset_urls[i]).content)


    def read_images(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, rows * cols)
            return images

    def read_labels(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            _, num_labels = struct.unpack('>II', f.read(8))
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            return labels

    
    def load_dataset(self) -> None:
        self.__train_x = dt.jtensors.JTensors(self.read_images(self.__dataset_path[0]))
        self.__train_y = dt.jtensors.JTensors(self.read_labels(self.__dataset_path[1]))
        self.__test_x = dt.jtensors.JTensors(self.read_images(self.__dataset_path[2]))
        self.__test_y = dt.jtensors.JTensors(self.read_labels(self.__dataset_path[3]))


    def train_data(self) -> Tuple[dt.jtensors.JTensors]:
        return self.__train_x, self.__train_y
    

    def test_data(self) -> Tuple[dt.jtensors.JTensors]:
        return self.__test_x, self.__test_y

