""" imports """
import autograd
from autograd.derivatives import (
    exp_deriv,
    log_deriv,
    matmul_deriv,
    sum_deriv,
    max_deriv,
    sqrt_deriv,
    tanh_deriv,
    dropout_deriv,
    from_list_deriv,
    to_list_deriv,
    unsqueeze_backward,
    squeeze_backward,
    reshape_backward,
    transpose_deriv
)
import numpy as np
from typing import Tuple


def transpose(tensor : autograd.jtensors.JTensors):

    return autograd.jtensors.JTensors(
        tensor().T,
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            transpose_deriv,
            "TransposeJBackward",
            tensor
        ) if tensor.require_grads else None
    )


def split(tensor : autograd.jtensors.JTensors, value : int | list[int]):

    return autograd.jtensors.JTensors(np.split(tensor(), value))


def zip(left : autograd.jtensors.JTensors, right : autograd.jtensors.JTensors, axis : float = 0):

    return autograd.jtensors.JTensors(np.stack((left(), right()), axis=axis))


def norm(tensor : autograd.jtensors.JTensors):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Tensor arg must be a JTensor"

    return sqrt(sum(tensor ** 2))


def sqrt(tensor : autograd.jtensors.JTensors):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Tensor arg must be a JTensor"

    return autograd.jtensors.JTensors(
        np.sqrt(tensor()),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            sqrt_deriv,
            "SqrtJBackward",
            tensor
        ) if tensor.require_grads else None
    )


def logsumexp(tensor : autograd.jtensors.JTensors):

    max = autograd.functionnal.max(tensor)
    ds = tensor - max
    sumOfExp = autograd.functionnal.exp(ds).max()
    return max + autograd.functionnal.log(sumOfExp)


def sigmoid(tensor : autograd.jtensors.JTensors):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Tensor arg must be a JTensor"

    return 1 / (1 + autograd.functionnal.exp(-1 * tensor))


def from_list(data : list[autograd.jtensors.JTensors]) -> autograd.jtensors.JTensors:

    return autograd.jtensors.JTensors(
        np.array(data),
        require_grads=True,
        operation=autograd.operations.CrossOperationBackward(
            from_list_deriv,
            "FromListJBackward",
            *data
        ) if any(list(filter(lambda x: (x.require_grads), data))) else None
    )


def to_list(tensor : autograd.jtensors.JTensors) -> list[autograd.jtensors.JTensors]:

    return [autograd.jtensors.JTensors(tensor[i], 
            require_grads=tensor.require_grads,
            operation=autograd.operations.CrossOperationBackward(
                to_list_deriv,
                "ToListJBackward",
                i, tensor
            )) for i in range(len(tensor))]


def dropout(tensor : autograd.jtensors.JTensors, p : float = 0.5):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Tensor arg must be a JTensor"

    mask = np.random.rand(*tensor.shape()) < p
    return autograd.jtensors.JTensors(
        np.where(mask, 0, tensor()),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            dropout_deriv,
            "DropoutJBackward",
            tensor, mask
        ) if tensor.require_grads else None
    )


def tanh(tensor : autograd.jtensors.JTensors):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Tensor arg must be a JTensor"

    return autograd.jtensors.JTensors(
        np.tanh(tensor()),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            tanh_deriv,
            "TanhJBackward",
            tensor
        ) if tensor.require_grads else None
    )


def softmax(tensor : autograd.jtensors.JTensors):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Tensor must be a JTensor"

    xp = exp(tensor)
    return xp / sum(xp)


def max(tensor : autograd.jtensors.JTensors, value : int | float):

    assert (isinstance(tensor, autograd.jtensors.JTensors)), "Invalid type of argument"
    assert (isinstance(value, (int, float))), "Value is not of scalar type"

    return autograd.jtensors.JTensors(
        np.maximum(value, tensor()),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            max_deriv,
            "MaxJBackward",
            tensor, value
        ) if tensor.require_grads else None
    )


def exp(tensor : autograd.jtensors.JTensors):

    return autograd.jtensors.JTensors(
        np.exp(tensor()),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            exp_deriv,
            "ExpJBackward",
            tensor
        ) if tensor.require_grads else None
    )


def log(tensor : autograd.jtensors.JTensors):

    return autograd.jtensors.JTensors(
        np.log(tensor()),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            log_deriv,
            "LogJBackward",
            tensor
        ) if tensor.require_grads else None
    )


def matmul(left : autograd.jtensors.JTensors, right : autograd.jtensors.JTensors):

    assert (isinstance(left, autograd.jtensors.JTensors) and isinstance(right, autograd.jtensors.JTensors)), "Invalid operand type for matmul"
    assert (left.shape()[-1] == right.shape()[0]), "Invalid shapes for matrices math multiplication. Shapes: " + str(left.shape()) + ", " + str(right.shape())

    require_grad : bool = (left.require_grads or right.require_grads)
    
    return autograd.jtensors.JTensors(
        np.matmul(left(), right()),
        require_grads=require_grad,
        operation=autograd.operations.CrossOperationBackward(
            matmul_deriv,
            "MmJBackward",
            left, right
        ) if require_grad else None
    )


def sum(tensor : autograd.jtensors.JTensors):

    return autograd.jtensors.JTensors(
        [np.sum(tensor())],
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            sum_deriv,
            "SumJBackward",
            tensor
        ) if tensor.require_grads else None
    )

""" Squeeze & Unsqueeze """

def squeeze(tensor : autograd.jtensors.JTensors, axis : int):

    return autograd.jtensors.JTensors(
        np.squeeze(tensor(), axis=axis),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            squeeze_backward,
            "SqueezeJBackward",
            tensor, axis
        ) if tensor.require_grads else None
    )


def unsqueeze(tensor : autograd.jtensors.JTensors, axis : int):

    return autograd.jtensors.JTensors(
        np.expand_dims(tensor(), axis=axis),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            unsqueeze_backward,
            "UnsqueezeJBackward",
            tensor, axis
        ) if tensor.require_grads else None
    )


def reshape(tensor : autograd.jtensors.JTensors, shape : Tuple[int]):

    #print("reshape shape", tensor.shape())
    #print("reshape stride", tensor.stride())
    #print("reshape shape after", tensor().reshape(shape).shape)
    #print("reshape stride after", tensor().reshape(shape).strides)
    return autograd.jtensors.JTensors(
        tensor().reshape(shape),
        require_grads=tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
            reshape_backward,
            "ReshapeJBackward",
            tensor, tensor.shape()
        )
    )


""" Basic tensors creation methods """

def random(*shape : int) -> autograd.jtensors.JTensors:

    """create a tensor of random values of shape 'shape'

    Returns:
        autograd.jtensors.JTensors: created tensor
    """

    return autograd.jtensors.JTensors(np.random.rand(*shape))


def ones(*shape : int, requires_grad : bool = False) -> autograd.jtensors.JTensors:

    """create a tensors filled with '1'

    Returns:
        JTensors: the created tensor
    """
    
    return autograd.jtensors.JTensors(np.ones(shape=shape), require_grads=requires_grad)


def zeros(*shape : int, requires_grad : bool = False) -> autograd.jtensors.JTensors:
    
    """create a tensors filled with '0'

    Returns:
        JTensors: the created tensor
    """

    return autograd.jtensors.JTensors(np.zeros(shape=shape), require_grads=requires_grad)


def zeros_like(tensor : autograd.jtensors.JTensors) -> autograd.jtensors.JTensors:
    
    """create a tensors filled with '0' and the same shape as 'tensor'

    Returns:
        JTensors: the created tensor
    """

    return autograd.jtensors.JTensors(np.zeros_like(tensor()), require_grads=tensor.require_grads)


def uniform_(_from : float | int, _to : float | int, size : int, require_grads : bool = False):

    return autograd.jtensors.JTensors(np.random.uniform(_from, _to, size), require_grads=require_grads)

