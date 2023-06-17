""" imports """
import dtorch.jtensors as jtensors
import dtorch.functionnal as fn
from typing import Tuple, Union
from abc import ABC, abstractmethod
import os

class Dataset:

    def __init__(self, x : jtensors.JTensors, y : jtensors.JTensors, ratio : float = 0.9, batch_size : int = 32, shuffle : bool = True) -> None:

        assert (len(x) == len(y)), "Invalid dataset shape"
        assert (len(x) % batch_size == 0), "Invalid batch size : " + str(batch_size) + ", for dataset of size : " + str(len(x))

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        batched_x = fn.split(x, len(x) // batch_size)
        batched_y = fn.split(y, len(y) // batch_size)
        data : jtensors.JTensors = fn.zip(batched_x, batched_y, axis=-3)
        if (shuffle):
            data.shuffle()
        self.__train : jtensors.JTensors = data[:int(len(data) * ratio)]
        self.__test : jtensors.JTensors = data[int(len(data) * ratio):]

    
    def get_all_data(self) -> Tuple[jtensors.JTensors, jtensors.JTensors]:
        """return all data

        Returns:
            Tuple[jtensors.JTensors, jtensors.JTensors]: (train, test)
        """

        return (self.__train, self.__test)


class JDataset(ABC):

    def __init__(self,
                 root : str = './data',
                 split : Union[Tuple[str], str] = ('train', 'test'),
                 download : bool = False) -> None: 
        self.root : str = root
        self.split : Union[Tuple[str], str] = split

        if not self._root_exists:
            os.makedirs(self.root)

        if download and not self._dataset_exists:
            self.download()
        else:
            assert (self._dataset_exists == True), "Can't find the dataset.\n You may want to download it by setting the `download` property to True."

        self.load_dataset()

    @property
    def _root_exists(self) -> bool:
        return os.path.isdir(self.root)


    @property
    def _dataset_exists(self) -> bool:
        raise NotImplementedError
    

    @property
    def data(self):
        raise NotImplementedError
    

    """ methods """

    """ public """

    @abstractmethod
    def download(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def load_dataset(self) -> None:
        raise NotImplementedError

