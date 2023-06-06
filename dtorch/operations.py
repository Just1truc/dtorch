""" imports """
import numpy as np
from typing import Any, Callable
import dtorch.jtensors
from dtorch.operation_class import Operation

class CrossOperationBackward(Operation):

    def __init__(self, deriv : Callable[..., None], operation_name : str, *args : dtorch.jtensors.JTensors | int | float) -> None:
        """_summary_

        Args:
            deriv (Callable[..., None]): the va arg is of form, (base_tensor, *tensors)
            operation_name (str): _description_
        """
        
        self.__deriv : Callable[..., None] = deriv
        self.__operation_name : str = operation_name
        self.__tensors : list[dtorch.jtensors.JTensors | int | float] = list(args)


    def __str__(self) -> str:
        return self.__operation_name
    

    def backward(self, base_tensor=None) -> Any:
        
        assert any(map(lambda x: x is not None, self.__tensors))

        self.__deriv(base_tensor, *self.__tensors)
