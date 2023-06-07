""" imports """
from typing import Any
import dtorch
import numpy as np

class Loss:

    def __init__(self) -> None:
        pass


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
        

class MSELoss:

    def __init__(self, reduction : str = 'mean') -> None:
        """mse loss class

        Args:
            reduction (str, optional): can be 'sum' | 'mean'. Defaults to 'mean'.
        """

        assert (reduction == 'mean' or reduction == 'sum'), "Invalid reduction provided"

        self.__reduction : str = reduction


    def __call__(self, outputs : dtorch.jtensors.JTensors, target : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:

        assert (outputs.shape == target.shape)

        #print("mse:", outputs.shape, outputs.stride)
        #print(((target - outputs) ** 2).stride)
        res : dtorch.jtensors.JTensors = dtorch.functionnal.sum((target - outputs) ** 2)
        if (self.__reduction == 'mean'):
            scl_nb = np.prod(outputs.shape)
            return res / int(scl_nb)
        return res
    

class BCELoss:

    def __init__(self, reduction : str = 'mean') -> None:
        
        assert (reduction == 'mean' or reduction == 'sum'), "Invalid reduction provided"

        self.__reduction = reduction


    def __call__(self, outputs : dtorch.jtensors.JTensors, targets : dtorch.jtensors.JTensors) -> Any:

        assert (np.array_equal(targets(), targets().astype(bool))), "Target tensors provided does not only contain binary classification"

        tot : dtorch.jtensors.JTensors = dtorch.functionnal.sum(targets * dtorch.functionnal.log(outputs) + (1 - targets) * dtorch.functionnal.log(1 - outputs))

        if (self.__reduction == 'mean'):
            return tot / len(outputs)
        return tot
        

# TODO : make LSE Loss


class BCEWithLogitsLoss:

    # TODO : using the logsumexp stabilization trick
    # rework

    def __init__(self, reduction : str = 'mean') -> None:
        
        assert (reduction == 'mean' or reduction == 'sum'), "Invalid reduction provided"

        self.__reduction = reduction


    def __call__(self, outputs_tensors : dtorch.jtensors.JTensors, targets : dtorch.jtensors.JTensors) -> Any:

        assert (np.array_equal(targets(), targets().astype(bool))), "Target tensors provided does not only contain binary classification"

        outputs : dtorch.jtensors.JTensors = outputs_tensors
        tot : dtorch.jtensors.JTensors = dtorch.functionnal.sum(targets * dtorch.functionnal.log(outputs) + (1 - targets) * dtorch.functionnal.log(1 - outputs))

        if (self.__reduction == 'mean'):
            return tot / len(outputs)
        return tot
