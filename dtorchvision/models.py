""" imports """
from typing import Any
import dtorch as dt
from abc import ABC
import os
import requests
from dtorch.nn import JPretrainedModel

class MNISTAutoEncoder_128_32(JPretrainedModel):

    def __init__(self, root : str = './models') -> None:
        """MNIST Autoencoder (1070 loss (MSE mean) on 60000 images)

        Args:
            root (str, optional): root of model folder. Defaults to '.models'.
        """

        super().__init__('mnist_128_32', 'dtorchvision', root=root)

        self.__model = AutoEncoder(784, [128, 32])
        self.__model.load(self.model_path)
        self.__model.eval()

    
    def forward(self, *args, **kwargs):
        return self.__model(*args, **kwargs)
    

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class AutoEncoder(dt.nn.Module):

    def __init__(self,
                input : int,
                hidden_sizes : list[int],
                dp : float = 0.0) -> None:
        """AutoEncoders

        Args:
            input (int): entry
            hidden_sizes (list[int]): list of hidden sizes I -> C (input to encoded space)
            dp (float, optional): dropout. Defaults to 0.0.
        """

        super().__init__()

        assert (len(hidden_sizes) > 0), "Invalid hidden size"

        layers = []
        last : int = input
        for i in range(len(hidden_sizes)):
            layers.append(dt.nn.Linear(last, hidden_sizes[i]))
            if (dp != 0.0):
                layers.append(dt.nn.Dropout(dp))
            layers.append(dt.nn.ReLU())
            last = hidden_sizes[i]
        for i in range(len(hidden_sizes) - 2, -1, -1):
            layers.append(dt.nn.Linear(last, hidden_sizes[i]))
            if (dp != 0.0):
                layers.append(dt.nn.Dropout(dp))
            layers.append(dt.nn.ReLU())
            last = hidden_sizes[i]
        layers.append(dt.nn.Linear(last, input))
        layers.append(dt.nn.ReLU())

        self.seq : dt.nn.Module = dt.nn.Sequential(
            *layers
        )

    
    def forward(self, x):
        return self.seq(x)