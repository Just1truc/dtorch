""" imports """
from typing import Tuple, Any, Optional
from dtorch.typing import types, DtAny

import numpy as np

import dtorch.operations
from dtorch.derivatives import *

""" Code """

class JTensors:

    """ Private """

    def __init__(self,
                 array : np.ndarray | list[float] | float,
                 require_grads : bool = False,
                 dtype : type | np.dtype = np.float64,
                 operation = None):

        assert (isinstance(array, (np.ndarray, list, float))), "Invalid parameter for argument of jtensor"

        if (isinstance(array, list)):
            self.__list : np.ndarray = np.array(array, dtype=dtype)
        elif (isinstance(array, float)):
            self.__list : np.ndarray = np.array([array], dtype=dtype)
        else:
            self.__list : np.ndarray = array
            self.__list = self.__list.astype(dtype)

        self.__operation : dtorch.operations.Operation = None
        if (require_grads):
            self.__operation : dtorch.operations.Operation = operation
        self.require_grads : bool = require_grads
        self.grad : JTensors = None
        self.__name : str = None


    def __call__(self):

        return self.__list


    def __getitem__(self, key : int):

        return self.__list[key]
    

    def __setitem__(self, key : int, value : float):

        self.__list[key] = value


    def __iter__(self):

        self.__i = 0
        return self


    def __next__(self):

        if self.__i < len(self.__list):
            self.__i += 1
            return self.__list[self.__i - 1]
        raise StopIteration
    

    def __sub__(self, other):

        assert (isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        res : JTensors | int | float = other
        if (isinstance(other, JTensors)):
            res = other.__list

        require_grad : bool =  self.require_grads or (isinstance(other, JTensors) and other.require_grads)

        return JTensors(
            self.__list - res if res is not None else self.__list,
            require_grads=require_grad,
            operation=dtorch.operations.CrossOperationBackward(sub_deriv, "SubJBackward", self, other)
            if require_grad else None
        )
    

    def __truediv__(self, other):

        assert (other is not None), "Can't divide by None"
        assert (isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        res : JTensors | int | float = other
        if (isinstance(other, JTensors)):
            res = other.__list

        require_grad : bool =  self.require_grads or (isinstance(other, JTensors) and other.require_grads)

        return JTensors(
            self.__list / res,
            require_grads=require_grad,
            operation=dtorch.operations.CrossOperationBackward(div_deriv, "DivJBackward", self, other)
            if require_grad else None
        )
    

    def __mul__(self, other):

        assert (other is not None), "Can't multiply by None (illogical)"

        assert (isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        res : JTensors | int | float = other
        if (isinstance(other, JTensors)):
            res = other.__list

        require_grad : bool = self.require_grads or (isinstance(other, JTensors) and other.require_grads)

        return JTensors(
            self.__list * res,
            require_grads=require_grad,
            operation=dtorch.operations.CrossOperationBackward(mul_deriv, "MulJBackward", self, other)
            if require_grad else None
        )
    

    def __add__(self, other):

        assert (other is None or isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        res : JTensors | int | float | None = other
        if (isinstance(other, JTensors)):
            res = other.__list

        require_grad : bool = self.require_grads or (isinstance(other, JTensors) and other.require_grads)

        return JTensors(
            self.__list + res if res is not None else self.__list,
            require_grads=require_grad,
            operation=dtorch.operations.CrossOperationBackward(add_deriv, "AddJBackward", self, other)
            if require_grad else None
        )
    

    def __rsub__(self, other):

        assert (other is None or isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"

        res : JTensors | int | float = other
        if (isinstance(other, JTensors)):
            res = other.__list

        require_grad : bool =  self.require_grads or (isinstance(other, JTensors) and other.require_grads)

        return JTensors(
            res - self.__list if res is not None else -self.__list,
            require_grads=require_grad,
            operation=dtorch.operations.CrossOperationBackward(sub_deriv, "SubJBackward", other, self)
            if require_grad else None
        )
    

    def __rtruediv__(self, other):

        assert (isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        res : JTensors | int | float = other
        if (isinstance(other, JTensors)):
            res = other.__list

        require_grad : bool =  self.require_grads or (isinstance(other, JTensors) and other.require_grads)

        return JTensors(
            res / self.__list,
            require_grads=require_grad,
            operation=dtorch.operations.CrossOperationBackward(div_deriv, "DivJBackward", other, self)
            if require_grad else None
        )
    

    def __rmul__(self, other):

        assert (isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        return self.__mul__(other)
    

    def __radd__(self, other):

        assert (other is None or isinstance(other, (JTensors, int, float))), "Invalid type for operation with jtensor"
        # consider operating by a direct value (int, float, ...)

        return self.__add__(other)


    def __str__(self) -> str:

        if self.require_grads:
            if self.__name is not None:
                return f"jtensor({self.__list}, .__operation = <{self.__operation}>, .__name = {self.__name})"
            return f"jtensor({self.__list}, .__operation = <{self.__operation}>)"
        if self.__name is not None:
            return f"jtensor({self.__list}, .__name = {self.__name})"
        return f"jtensor({self.__list})"
 

    def __repr__(self) -> str:

        return self.__str__()
    

    def __pow__(self, factor : int):

        assert (isinstance(factor, int)), "Invalid factor for power"

        return JTensors(
            np.power(self.__list, factor),
            require_grads=self.require_grads,
            operation=dtorch.operations.CrossOperationBackward(pow_deriv, "PowJBackward", self, factor)
            if self.require_grads else None
        )


    def __len__(self):
        return self.__list.__len__()


    """ Public """
    def backward(self, base_tensor = None, forced : bool = False) -> None:
        """Backpropagate through the tensors
        """
        assert (base_tensor is None or isinstance(base_tensor, JTensors)), "the tensor given as argument is not a tensor"
        assert (self.require_grads == True), "The tensor require_grads argument is required to be set to true for .backward to be called"
        assert (forced or np.array(self.__list.shape).all()), "Backward method can only be called on scalar input"

        if (base_tensor is None):
            base_tensor = JTensors([1])
            
        if (self.__operation is not None):
            self.__operation.backward(base_tensor)


    @property
    def ndims(self) -> int:

        """number of dimensions of the tensor

        Returns:
            int: the number of dims
        """

        return len(self.__list.shape)
    

    @property
    def dtype(self) -> np.dtype:

        """return the dtype of the tensor

        Returns:
            np.dtype: the dtype
        """

        return self.__list.dtype
    

    @dtype.setter
    def dtype(self, dtype : np.dtype) -> None:

        """set the dtype of the tensor

        Args:
            dtype (np.dtype): the dtype to set
        """

        self.__list = self.__list.astype(dtype)
    

    @property
    def T(self) -> Any:
        
        """return the transpose of the tensor

        Returns:
            Any: the transpose
        """

        return dtorch.functionnal.transpose(self)


    @property
    def itemsize(self) -> int:

        """return the itemsize of the tensor

        Returns:
            int: the itemsize
        """

        return self.__list.itemsize
    

    @property
    def size(self) -> int:

        """number of elements in the tensor

        Returns:
            int: the number of elements
        """

        return self.__list.size


    def isLeaf(self) -> bool:
        """
        """

        return self.__operation == None


    def numpy(self):
        """return the numpy array
        """

        return self.__list


    @property
    def shape(self) -> Tuple[int]:
        """return the shape of the tensor

        Returns:
            Tuple[int]: tensor's shape
        """

        return self.__list.shape
    
    
    def transpose(self) -> Any:

        return dtorch.functionnal.transpose(self)
    

    def reshape(self, *shape) -> Any:
        """reshape the tensor
        """

        return dtorch.functionnal.reshape(self, shape)
    

    @property
    def stride(self) -> Tuple[int]:
        """return the stride of the tensor

        Returns:
            Tuple[int]: tensor's stride
        """

        return self.__list.strides

    
    def update(self, new_list : np.ndarray) -> None:

        """update wrapped list
        """

        self.__list = new_list


    def add_name(self, name : str) -> None:
            
        """add a name to the tensor
        """

        self.__name = name


    def get_name(self) -> str:
            
        """get the name of the tensor
        """

        return self.__name
    
    def max(self) -> Any:

        return JTensors(
            [self.__list.max()],
            require_grads=self.require_grads,
            operation=dtorch.operations.CrossOperationBackward(
                real_max_deriv,
                "MaxJBackward1",
                self
            ) if self.require_grads else None
        )


    def sum(self, axis : Optional[Tuple[int]] = None, keepdims : bool = False) -> Any:

        return dtorch.functionnal.sum(self, axis, keepdims)
    

    def rearrange(self, pattern : str) -> Any:

        """rearrange the tensor

        Returns:
            Any: the rearranged tensor
        """

        return dtorch.einops.rearrange(self, pattern)


    def shuffle(self) -> Any:

        np.random.shuffle(self.__list)
        return self


    def norm(self) -> Any:

        """compute the norm of the tensor

        Returns:
            JTensors: a tensor with the norm
        """

        return dtorch.functionnal.norm(self)
    

    def detach(self) -> Any:

        """detach the tensor\n
        return a new tensor with the same value but no gradient
        """

        return JTensors(
            self.__list,
            require_grads=False,
            operation=None
        )


    def clone(self) -> Any:

        """clone the tensor
        """
            
        return JTensors(
            self.__list.copy(),
            require_grads=self.require_grads,
            operation=self.__operation
        )


    def unsqueeze(self, dim : int):

        """insert a dimension

        Returns:
            jensor: jtensors with a new dimension
        """

        return dtorch.functionnal.unsqueeze(self, dim)


    def squeeze(self, dim : int):

        """remove a dimension

        Returns:
            jensor: jtensors with a removed dimension
        """

        return dtorch.functionnal.squeeze(self, dim)
