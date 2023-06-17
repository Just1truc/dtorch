""" imports """
from typing import Tuple
from dtorch.dataset import JDataset
import os
import requests
import numpy as np
import dtorch as dt

class IrisDataset(JDataset):

    def __init__(self, root: str = './data', split: Tuple[str] | str = 'train', download: bool = False) -> None:

        self.__dataset_url : str = 'https://raw.githubusercontent.com/Just1truc/dtorch/main/datasets/iris_dataset.csv'
        self.__dataset_path : str = os.path.join(root, 'iris_dataset.csv')
        self.__data = None

        super().__init__(root, split, download)


    @property
    def _dataset_exists(self) -> bool:
        return os.path.isfile(self.__dataset_path)
    

    @property
    def data(self):
        return self.__data
    

    """ methods """

    """ public """

    def download(self) -> None:
        with open(self.__dataset_path, 'wb') as f:
            f.write(requests.get(self.__dataset_url).content)


    def load_dataset(self) -> None:
        x, y = (np.loadtxt(self.__dataset_path, skiprows=1, delimiter=',', usecols=(0, 1, 2, 3)).T, np.loadtxt(self.__dataset_path, skiprows=1, delimiter=',', usecols=(4,), dtype='<U14'))
        self.__data = (dt.tensor(x), dt.tensor(y, dtype=y.dtype))
