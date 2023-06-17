""" imports """
import dtorch
from dtorch.derivatives import (
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
    transpose_deriv,
    as_strided_deriv
)
import numpy as np
from typing import Tuple
from dtorch.typing import types, Optional, DtOptional
import math
import dtorch as dt

@types(tensor = dtorch.jtensors.JTensors, 
       axis = DtOptional(Tuple),
       return_type=dtorch.jtensors.JTensors)
def transpose(tensor : dtorch.jtensors.JTensors, axis : Optional[Tuple[int, int]] = (1, 0)) -> dtorch.jtensors.JTensors:

    assert (len(axis) == 2 or len(axis) == tensor.shape)

    return dtorch.jtensors.JTensors(
        np.transpose(tensor, axes=axis),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            transpose_deriv,
            "TransposeJBackward",
            tensor, axis
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
       value = (int, list),
       return_type=dtorch.jtensors.JTensors)
def split(tensor : dtorch.jtensors.JTensors, value : int | list[int]) -> dtorch.jtensors.JTensors:

    """ No backward support yet """

    return dtorch.jtensors.JTensors(np.split(tensor(), value))


@types(left = dtorch.jtensors.JTensors,
       right = dtorch.jtensors.JTensors,
       axis = int,
       return_type=dtorch.jtensors.JTensors)
def zip(left : dtorch.jtensors.JTensors, right : dtorch.jtensors.JTensors, axis : int = 0) -> dtorch.jtensors.JTensors:

    """ No backward support yet """

    return dtorch.jtensors.JTensors(np.stack((left(), right()), axis=axis))


@types(tensor = dtorch.jtensors.JTensors,
       return_type=dtorch.jtensors.JTensors)
def norm(tensor : dtorch.jtensors.JTensors): return sqrt(sum(tensor ** 2))


@types(tensor = dtorch.jtensors.JTensors,
       return_type=dtorch.jtensors.JTensors)
def sqrt(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:

    return dtorch.jtensors.JTensors(
        np.sqrt(tensor()),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            sqrt_deriv,
            "SqrtJBackward",
            tensor
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
        return_type=dtorch.jtensors.JTensors)
def logsumexp(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:

    max = dtorch.functionnal.max(tensor)
    ds = tensor - max
    sumOfExp = dtorch.functionnal.exp(ds).max()
    return max + dtorch.functionnal.log(sumOfExp)


@types(tensor = dtorch.jtensors.JTensors,
         return_type=dtorch.jtensors.JTensors)
def sigmoid(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:

    return 1 / (1 + dtorch.functionnal.exp(-1 * tensor))


@types(data = list,
         return_type=dtorch.jtensors.JTensors)
def from_list(data : list[dtorch.jtensors.JTensors]) -> dtorch.jtensors.JTensors:

    return dtorch.jtensors.JTensors(
        np.array(data),
        require_grads=True,
        operation=dtorch.operations.CrossOperationBackward(
            from_list_deriv,
            "FromListJBackward",
            *data
        ) if any(list(filter(lambda x: (x.require_grads), data))) else None
    )


@types(tensor = dtorch.jtensors.JTensors,
        return_type=list)
def to_list(tensor : dtorch.jtensors.JTensors) -> list[dtorch.jtensors.JTensors]:

    return [dtorch.jtensors.JTensors(tensor[i], 
            require_grads=tensor.require_grads,
            operation=dtorch.operations.CrossOperationBackward(
                to_list_deriv,
                "ToListJBackward",
                i, tensor
            )) for i in range(len(tensor))]


@types(tensor = dtorch.jtensors.JTensors,
       shape = Tuple,
       strides = Tuple,
       return_type=dtorch.jtensors.JTensors)
def as_strided(tensor : dtorch.jtensors.JTensors, shape : Tuple[int, int], strides : Tuple[int, int]) -> dtorch.jtensors.JTensors:

    """change the stride of a tensor

    may support backward.

    Returns:
        jtensor: result
    """

    stride = tuple(np.array(strides) * tensor.itemsize)
    return dtorch.jtensors.JTensors(
        np.lib.stride_tricks.as_strided(tensor(), shape=shape, strides=stride),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            as_strided_deriv,
            "AsStridedJBackward",
            tensor, shape, stride
        ) if tensor.require_grads else None
    )


@types(input = dtorch.jtensors.JTensors,
       weight = dtorch.jtensors.JTensors,
       bias = DtOptional(dtorch.jtensors.JTensors),
       stride = DtOptional(int),
       return_type = dtorch.jtensors.JTensors)
def conv1d(input : dtorch.jtensors.JTensors,
           weight : dtorch.jtensors.JTensors,
           bias : Optional[dtorch.jtensors.JTensors] = None,
           stride : Optional[int] = 1) -> dtorch.jtensors.JTensors:
    """convolution of a 1d tensor

    Args:
        input (dtorch.jtensors.JTensors): 1d tensor (batch_size, in_channel, width) or (in_channel, width)
        weight (dtorch.jtensors.JTensors): (out_channel, in_channel, kernel_width) or (in_channel, kernel_width)
        bias (dtorch.jtensors.JTensors, optional): (out_channel). Defaults to None.
        stride (int, optional): *movement speed* of the kernel. Defaults to 1.

    Returns:
        dtorch.jtensors.JTensors: convolution result
    """

    # TODO : Take into account batch missing possibility
    # TODO : Test batched, multiple channel, ect
    # Take into account the stride also

    assert (input.ndims == 2 or input.ndims == 3) and (weight.ndims == 2 or weight.ndims == 3), "input and weight must be 1d or 2d"
    assert (bias is None) or (bias.ndims == 1), "bias must be 1d"
    b, ic, w = input.shape
    oc, ic_, kw = weight.shape
    assert ic == ic_, "input and weight must have the same in_channel"
    assert (w - kw) % stride == 0, "width of input and weight must be compatible"
    assert (bias is None) or (oc == bias.shape[0]), "bias and weight must have the same out_channel"
    assert (w >= kw), "width of input must be greater than width of weight"

    new_w = w - (kw - 1) * stride
    strided_input = as_strided(input, (b, 1, new_w, ic, kw), (w, 0, 1, ic * kw * stride, 1))

    x : dtorch.jtensors.JTensors = strided_input * weight.unsqueeze(1)
    x = x.sum(axis=(3, 4))

    if bias is not None:
        x = x + bias

    return x


@types(input = dtorch.jtensors.JTensors,
       weight = dtorch.jtensors.JTensors,
       bias = DtOptional(dtorch.jtensors.JTensors),
       stride = DtOptional(int),
       return_type = dtorch.jtensors.JTensors)
def conv2d(input : dtorch.jtensors.JTensors,
           weight : dtorch.jtensors.JTensors,
           bias : Optional[dtorch.jtensors.JTensors] = None,
           stride : int = 1) -> dtorch.jtensors.JTensors:
    
    """convolution of a 2d tensor

    Args:
        input (dtorch.jtensors.JTensors): 2d tensor (batch_size, in_channel, height, width) or (in_channel, height, width)
        weight (dtorch.jtensors.JTensors): (out_channel, in_channel, kernel_height, kernel_width) or (in_channel, kernel_height, kernel_width)
        bias (dtorch.jtensors.JTensors, optional): (out_channel). Defaults to None.
        stride (int, optional): *movement speed* of the kernel. Defaults to 1.

    Returns:
        dtorch.jtensors.JTensors: convolution result
    """

    assert (input.ndims == 3 or input.ndims == 4) and (weight.ndims == 3 or weight.ndims == 4), "input and weight must be 2d or 3d"
    assert (bias is None) or (bias.ndims == 1), "bias must be 1d"
    if (input.ndims == 4):
        b, ic, h, w = input.shape
    else:
        ic, h, w = input.shape
    oc, ic_, kh, kw = weight.shape
    assert ic == ic_, "input and weight must have the same in_channel"
    assert (h - kh) % stride == 0 and (w - kw) % stride == 0, "height and width of input and weight must be compatible"
    assert (w >= kw), "width of input must be greater or equal to the width of weight"

    new_w, new_h = (w - (kw - 1) * stride, h - (kh - 1) * stride)
    #new_shape = (b, new_h, new_w, ic, kh, kw) if input.ndims == 4 else (new_h, new_w, ic, kh, kw)
    #new_stride = (w * h * ic, kw, 1, w * h, kw, 1) if input.ndims == 4 else (kw, 1, w * h, kw, 1)
    new_shape = (b, 1, new_h, new_w, ic, kh, kw) if input.ndims == 4 else (1, new_h, new_w, ic, kh, kw)
    new_stride = (w * h * ic, 1, kw, 1, w * h, kw, 1) if input.ndims == 4 else (1, kw, 1, w * h, kw, 1)
    strided_input = as_strided(input, new_shape, new_stride)
    #print(strided_input)
    y = strided_input * weight.reshape(oc, 1, 1, ic, kh, kw)
    y = y.sum(axis=(4, 5, 6))
    if bias is not None:
        y = y + bias.reshape(oc, 1, 1)
    return y


@types(tensor = dtorch.jtensors.JTensors,
       p = float,
       return_type=dtorch.jtensors.JTensors)
def dropout(tensor : dtorch.jtensors.JTensors, p : float = 0.5):

    mask = np.random.rand(*tensor) < p
    return dtorch.jtensors.JTensors(
        np.where(mask, 0, tensor()),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            dropout_deriv,
            "DropoutJBackward",
            tensor, mask
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
         return_type=dtorch.jtensors.JTensors)
def tanh(tensor : dtorch.jtensors.JTensors):

    return dtorch.jtensors.JTensors(
        np.tanh(tensor()),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            tanh_deriv,
            "TanhJBackward",
            tensor
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
         return_type=dtorch.jtensors.JTensors)
def softmax(tensor : dtorch.jtensors.JTensors):

    xp = exp(tensor)
    return xp / sum(xp)


@types(tensor = dtorch.jtensors.JTensors,
       value = int | float,
        return_type=dtorch.jtensors.JTensors)
def max(tensor : dtorch.jtensors.JTensors, value : int | float):

    return dtorch.jtensors.JTensors(
        np.maximum(value, tensor()),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            max_deriv,
            "MaxJBackward",
            tensor, value
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
        return_type=dtorch.jtensors.JTensors)
def exp(tensor : dtorch.jtensors.JTensors):

    return dtorch.jtensors.JTensors(
        np.exp(tensor()),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            exp_deriv,
            "ExpJBackward",
            tensor
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
        return_type=dtorch.jtensors.JTensors)
def log(tensor : dtorch.jtensors.JTensors):

    return dtorch.jtensors.JTensors(
        np.log(tensor()),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            log_deriv,
            "LogJBackward",
            tensor
        ) if tensor.require_grads else None
    )


@types(left = dtorch.jtensors.JTensors,
       right = dtorch.jtensors.JTensors,
       return_type=dtorch.jtensors.JTensors) 
def matmul(left : dtorch.jtensors.JTensors, right : dtorch.jtensors.JTensors):

    require_grad : bool = (left.require_grads or right.require_grads)
    
    return dtorch.jtensors.JTensors(
        np.matmul(left(), right()),
        require_grads=require_grad,
        operation=dtorch.operations.CrossOperationBackward(
            matmul_deriv,
            "MmJBackward",
            left, right
        ) if require_grad else None
    )


@types(tensor = dtorch.jtensors.JTensors,
       axis = DtOptional(Tuple),
       keepdims = bool,
       return_type=dtorch.jtensors.JTensors)
def sum(tensor : dtorch.jtensors.JTensors, axis : Optional[Tuple[int]] = None, keepdims : bool = False):

    # TODO : add tests on axis sum

    return dtorch.jtensors.JTensors(
        np.sum(tensor(), keepdims=keepdims) if axis is None else np.sum(tensor(), axis=axis, keepdims=keepdims),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            sum_deriv,
            "SumJBackward",
            tensor, axis
        ) if tensor.require_grads else None
    )

""" Squeeze & Unsqueeze """

@types(tensor = dtorch.jtensors.JTensors,
        axis = int,
        return_type=dtorch.jtensors.JTensors)
def squeeze(tensor : dtorch.jtensors.JTensors, axis : int):

    return dtorch.jtensors.JTensors(
        np.squeeze(tensor(), axis=axis),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            squeeze_backward,
            "SqueezeJBackward",
            tensor, axis
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
        axis = int,
        return_type=dtorch.jtensors.JTensors)
def unsqueeze(tensor : dtorch.jtensors.JTensors, axis : int):

    return dtorch.jtensors.JTensors(
        np.expand_dims(tensor(), axis=axis),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            unsqueeze_backward,
            "UnsqueezeJBackward",
            tensor, axis
        ) if tensor.require_grads else None
    )


@types(tensor = dtorch.jtensors.JTensors,
        shape = Tuple,
        return_type=dtorch.jtensors.JTensors)
def reshape(tensor : dtorch.jtensors.JTensors, shape : Tuple[int]):

    return dtorch.jtensors.JTensors(
        tensor().reshape(shape),
        require_grads=tensor.require_grads,
        operation=dtorch.operations.CrossOperationBackward(
            reshape_backward,
            "ReshapeJBackward",
            tensor, tensor.shape
        )
    )


""" Basic tensors creation methods """

@types(start = int, end = int, step = int,
        return_type=dtorch.jtensors.JTensors)
def arange(start : int, end : int, step : int = 1) -> dtorch.jtensors.JTensors:

    """create a tensor of values from 'start' to 'end' with a step of 'step'

    Args:
        start (int): start value
        end (int): end value
        step (int, optional): step between values. Defaults to 1.

    Returns:
        JTensors: the created tensor
    """

    return dtorch.jtensors.JTensors(np.arange(start, end, step))


def tensor(list : list | np.ndarray, require_grads : bool = False, dtype : type | np.dtype = np.float64) -> dtorch.jtensors.JTensors:

    """create a tensor from a list or a numpy array

    Args:
        list (list | np.ndarray): the list or numpy array to create the tensor from
    """

    return dtorch.jtensors.JTensors(list, require_grads, dtype)


def random(*shape : int) -> dtorch.jtensors.JTensors:

    """create a tensor of random values of shape 'shape'

    Returns:
        autograd.jtensors.JTensors: created tensor
    """

    return dtorch.jtensors.JTensors(np.random.rand(*shape))


def ones(*shape : int, requires_grad : bool = False) -> dtorch.jtensors.JTensors:

    """create a tensors filled with '1'

    Returns:
        JTensors: the created tensor
    """
    
    return dtorch.jtensors.JTensors(np.ones(shape=shape), require_grads=requires_grad)


def zeros(*shape : int, requires_grad : bool = False) -> dtorch.jtensors.JTensors:
    
    """create a tensors filled with '0'

    Returns:
        JTensors: the created tensor
    """

    return dtorch.jtensors.JTensors(np.zeros(shape=shape), require_grads=requires_grad)


@types(nb_feat = int, return_type = dtorch.jtensors.JTensors, size = int, require_grads = DtOptional(bool))
def xavier(nb_feat : int, size : int, require_grads : Optional[bool] = False) -> dtorch.jtensors.JTensors:

    """xavier initialisation function

    Returns:
       JTensors : initialized tensor
    """

    stdv = 1 / math.sqrt(nb_feat)
    return uniform_(-stdv, stdv, size, require_grads)


@types(tensor = dtorch.jtensors.JTensors,
        return_type=dtorch.jtensors.JTensors)
def zeros_like(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:
    
    """create a tensors filled with '0' and the same shape as 'tensor'

    Returns:
        JTensors: the created tensor
    """

    return dtorch.jtensors.JTensors(np.zeros_like(tensor()), require_grads=tensor.require_grads)


@types(_from = float | int, _to = float | int, size = int, requires_grad = bool,
        return_type=dtorch.jtensors.JTensors)
def uniform_(_from : float | int, _to : float | int, size : int, require_grads : bool = False):

    return dtorch.jtensors.JTensors(np.random.uniform(_from, _to, size), require_grads=require_grads)

