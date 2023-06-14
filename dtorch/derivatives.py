""" imports """
import dtorch.jtensors
import numpy as np
from typing import Tuple


def as_strided_deriv(base_tensor, *tensors):

    # TODO: fix this

    tensor : dtorch.jtensors.JTensors = tensors[0]
    shape : Tuple[int] = tensors[1]
    strides : Tuple[int] = tensors[2]

    if (tensor.require_grads):
        indexes = np.lib.stride_tricks.as_strided(np.arange(tensor().size), shape=shape, strides=strides)
        res = np.bincount(indexes.ravel(), base_tensor().ravel())
        res.resize(tensor.shape)
        restrided_base_tensor = np.lib.stride_tricks.as_strided(base_tensor(), shape=tensor.shape, strides=tensor.stride)
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(res)
        else:
            tensor.grad = dtorch.jtensors.JTensors(res)
        tensor.backward(tensor.grad, forced=True)

""" Complex """

def unsqueeze_backward(base_tensor, *tensors):
    
    tensor : dtorch.jtensors.JTensors = tensors[0]
    dim : int = tensors[1]

    if (tensor.require_grads):
        grad = dtorch.jtensors.JTensors(np.squeeze(np.sum(base_tensor(), axis=dim, keepdims=True), axis=dim))
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


def squeeze_backward(base_tensor, *tensors):
    
    tensor : dtorch.jtensors.JTensors = tensors[0]
    dim : int = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(np.expand_dims(base_tensor(), axis=dim))
        else:
            tensor.grad = dtorch.jtensors.JTensors(np.expand_dims(base_tensor(), axis=dim))
        tensor.backward(tensor.grad, forced=True)


def reshape_backward(base_tensor, *tensors):
    
    tensor : dtorch.jtensors.JTensors = tensors[0]
    old_shape : Tuple[int] = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(base_tensor().reshape(old_shape))
        else:
            tensor.grad = dtorch.jtensors.JTensors(base_tensor().reshape(old_shape))
        tensor.backward(tensor.grad, forced=True)


""" op """

def sqrt_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(base_tensor() * (1 / (2 * np.sqrt(tensor()))))
        else:
            tensor.grad = dtorch.jtensors.JTensors(base_tensor() * (1 / (2 * np.sqrt(tensor()))))
        tensor.backward(tensor.grad, forced=True)


def real_max_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        grad = dtorch.jtensors.JTensors(
            base_tensor() * np.where(tensor() == tensor().max(), 1, 0)
        )
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


def max_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]
    value : float | int = tensors[1]

    if (tensor.require_grads):
        grad = dtorch.jtensors.JTensors(
            np.where(tensor() < value, 0, base_tensor())
        )
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


def transpose_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        grad = dtorch.jtensors.JTensors(
            np.transpose(base_tensor())
        )
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


# just a matrix on what parameters depend on
def matmul_deriv(base_tensor, *tensors):
    left : dtorch.jtensors.JTensors = tensors[0]
    right : dtorch.jtensors.JTensors = tensors[1]

    if (left.require_grads):
        if (left.isLeaf()):
            left.grad += dtorch.jtensors.JTensors(np.matmul(base_tensor(), np.swapaxes(right(), -1, -2)).reshape(left.shape))
        else:
            left.grad = dtorch.jtensors.JTensors(np.matmul(base_tensor(), np.swapaxes(right(), -2, -1)).reshape(left.shape))
        left.backward(left.grad, forced=True)

    if (right.require_grads):
        if (right.isLeaf()):
            right.grad += dtorch.jtensors.JTensors(np.matmul(np.swapaxes(left(), -1, -2), base_tensor()).reshape(right.shape))
        else:
            right.grad = dtorch.jtensors.JTensors(np.matmul(np.swapaxes(left(), -1, -2), base_tensor()).reshape(right.shape))
        right.backward(right.grad, forced=True)


def sum_deriv(base_tensor, *tensors):
    from_tensor : dtorch.jtensors.JTensors = tensors[0]
    axis : Tuple[int] = tensors[1]

    if (from_tensor.require_grads):
        #tensor = dtorch.jtensors.JTensors(np.lib.stride_tricks.as_strided(np.ones(from_tensor.shape), shape=from_tensor.shape, strides=from_tensor.stride))
        k = base_tensor.numpy().copy()
        if (axis is not None):
            o = np.array(from_tensor.shape)
            o[list(axis)] = np.ones(len(axis), dtype=int)
            k = k.reshape(tuple(o))
        k = np.broadcast_to(k, from_tensor.shape)
        if (from_tensor.isLeaf()):
            from_tensor.grad += dtorch.jtensors.JTensors(k)
        else:
            from_tensor.grad = dtorch.jtensors.JTensors(k)
        from_tensor.backward(from_tensor.grad, forced=True)


""" Basics """

def exp_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(base_tensor() * np.exp(tensor()))
        else:
            tensor.grad = dtorch.jtensors.JTensors(base_tensor() * np.exp(tensor()))
        tensor.backward(tensor.grad, forced=True)


def log_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(base_tensor() *  (1 / (tensor())))
        else:
            tensor.grad = dtorch.jtensors.JTensors(base_tensor() *  (1 / (tensor())))
        tensor.backward(tensor.grad, forced=True)


def dimensionToSum(left : Tuple[int], right : Tuple[int]) -> Tuple[int]:

    left = list(left)
    right = list(right)
    r = 0
    l = 0
    to_sum_left = []

    while (l < len(left)):
        if (r >= len(right)):
            l += 1
            to_sum_left.append(len(left) - l)
            continue
        if (right[-1 - r] == 1):
            r += 1
            l += 1
            to_sum_left.append(len(left) - l)
            continue
        if (left[-1 -l] == 1):
            l += 1
            r += 1
            continue
        if (left[-1 -l] == right[-1 -r]):
            r += 1
            l += 1
            continue
        l += 1
        r += 1
        to_sum_left.append(len(left) - l)

    return tuple(to_sum_left)


def mul_deriv(base_tensor, *tensors):

    unpacked_tensors : list[dtorch.jtensors.JTensors] = list(tensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], dtorch.jtensors.JTensors)
    #if (is_tensor):
    #    left_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[0].shape)))
    #    right_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[1].shape)))
    
    #print("pouete", base_tensor.shape)
    #print("proutgot", unpacked_tensors[0].shape, unpacked_tensors[1].shape if is_tensor else unpacked_tensors[1])
    if (unpacked_tensors[0].require_grads):
        right_data = unpacked_tensors[1]() if is_tensor else unpacked_tensors[1]
        #grad = dtorch.jtensors.JTensors((np.sum(right_data * base_tensor(), axis=dimensionToSum((right_data * base_tensor()).shape, unpacked_tensors[0].shape)).reshape(*unpacked_tensors[0].shape)
        #                                                        if is_tensor and left_size < right_size
        #                                                        else (right_data * base_tensor.numpy()).reshape(*unpacked_tensors[0].shape)))
        axis = dimensionToSum((right_data * base_tensor()).shape, unpacked_tensors[0].shape)
        grad = dtorch.jtensors.JTensors(np.sum(right_data * base_tensor(), axis=axis).reshape(*unpacked_tensors[0].shape))
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += grad
        else:
            unpacked_tensors[0].grad = grad
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        #grad = dtorch.jtensors.JTensors((np.sum(unpacked_tensors[0]() * base_tensor(), axis=dimensionToSum((unpacked_tensors[0]() * base_tensor()).shape, unpacked_tensors[1].shape)).reshape(*unpacked_tensors[1].shape)
        #                                                       if right_size < left_size
        #                                                    else (base_tensor.numpy() * unpacked_tensors[0]()).reshape(*unpacked_tensors[1].shape)))
        axis = dimensionToSum((unpacked_tensors[0]() * base_tensor()).shape, unpacked_tensors[1].shape)
        grad = dtorch.jtensors.JTensors(np.sum(unpacked_tensors[0]() * base_tensor(), axis=axis).reshape(*unpacked_tensors[1].shape))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def tanh_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(base_tensor() * (1 - np.tanh(tensor()) ** 2))
        else:
            tensor.grad = dtorch.jtensors.JTensors(base_tensor() * (1 - np.tanh(tensor()) ** 2))
        tensor.backward(tensor.grad, forced=True)


def to_list_deriv(base_tensor, *tensors):

    index : int = tensors[0]
    tensor : dtorch.jtensors.JTensors = tensors[1]

    if (tensor.require_grads):
        if (tensor.grad is None):
            tensor.grad = dtorch.jtensors.JTensors(np.zeros(tensor.shape))
        if (tensor.isLeaf()):
            tensor.grad[index] += base_tensor()
        else:
            tensor.grad[index] = base_tensor()
        tensor.backward(tensor.grad, forced=True)


def dropout_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]
    mask : dtorch.jtensors.JTensors = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(np.where(mask, 0, base_tensor.numpy()))
        else:
            tensor.grad = dtorch.jtensors.JTensors(np.where(mask, 0, base_tensor.numpy()))
        tensor.backward(tensor.grad, forced=True)


def from_list_deriv(base_tensor, *tensors):

    tensors : list[dtorch.jtensors.JTensors] = list(tensors)

    for i in range(len(tensors) - 1, -1, -1):
        if (tensors[i].require_grads):
            if (tensors[i].isLeaf()):
                tensors[i].grad += dtorch.jtensors.JTensors(base_tensor()[i])
            else:
                tensors[i].grad = dtorch.jtensors.JTensors(base_tensor()[i])
            tensors[i].backward(tensors[i].grad, forced=True)


# def add_deriv(base_tensor, *tensors):

#     unpacked_tensors : list[dtorch.jtensors.JTensors] = list(tensors)
#     is_tensor : bool = isinstance(unpacked_tensors[1], dtorch.jtensors.JTensors)
#     left_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[0].shape)))
#     right_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[1].shape)))

#     if (unpacked_tensors[0].require_grads):
#         grad = dtorch.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[1].shape, unpacked_tensors[0].shape)).reshape(*unpacked_tensors[0].shape)
#                                                                 if left_size < right_size
#                                                                 else base_tensor.numpy().reshape(*unpacked_tensors[0].shape))
#         if (unpacked_tensors[0].isLeaf()):
#             unpacked_tensors[0].grad += grad
#         else:
#             unpacked_tensors[0].grad = grad
#         unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

#     if (is_tensor and unpacked_tensors[1].require_grads):
#         grad = dtorch.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[0].shape, unpacked_tensors[1].shape)).reshape(*unpacked_tensors[1].shape)
#                                                                if right_size < left_size
#                                                                else base_tensor.numpy().reshape(*unpacked_tensors[1].shape))
#         if (unpacked_tensors[1].isLeaf()):
#             unpacked_tensors[1].grad += grad
#         else:
#             unpacked_tensors[1].grad = grad
#         unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)
def add_deriv(base_tensor, *tensors):

    unpacked_tensors : list[dtorch.jtensors.JTensors] = list(tensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], dtorch.jtensors.JTensors)
    
    if (unpacked_tensors[0].require_grads):
        axis = dimensionToSum(base_tensor().shape, unpacked_tensors[0].shape)
        grad = dtorch.jtensors.JTensors(np.sum(base_tensor(), axis=axis).reshape(*unpacked_tensors[0].shape))
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += grad
        else:
            unpacked_tensors[0].grad = grad
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        grad = dtorch.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[0].shape, unpacked_tensors[1].shape)).reshape(*unpacked_tensors[1].shape))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)



def div_deriv(base_tensor, *tensors):

    unpacked_tensors : list[dtorch.jtensors.JTensors] = list(tensors)

    zis_tensor : bool = isinstance(unpacked_tensors[0], dtorch.jtensors.JTensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], dtorch.jtensors.JTensors)

    # prout
    if (zis_tensor and unpacked_tensors[0].require_grads):
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += dtorch.jtensors.JTensors(base_tensor() * (1 / (unpacked_tensors[1]() if is_tensor else unpacked_tensors[1])))
        else:
            unpacked_tensors[0].grad = dtorch.jtensors.JTensors(base_tensor() * (1 / (unpacked_tensors[1]() if is_tensor else unpacked_tensors[1])))
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        grad = dtorch.jtensors.JTensors(-base_tensor() * ((unpacked_tensors[0]() if zis_tensor else unpacked_tensors[0]) / np.power(unpacked_tensors[1](), 2)))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def sub_deriv(base_tensor, *tensors):
    # need to be fixed

    unpacked_tensors : list[dtorch.jtensors.JTensors] = list(tensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], dtorch.jtensors.JTensors)
    left_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[0].shape)))
    right_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[1].shape)))

    if (unpacked_tensors[0].require_grads):
        grad = dtorch.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[1].shape, unpacked_tensors[0].shape)).reshape(*unpacked_tensors[0].shape)
                                                                if left_size < right_size
                                                                else base_tensor.numpy().reshape(*unpacked_tensors[0].shape))
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += grad
        else:
            unpacked_tensors[0].grad = grad
        # probleme diagnostiqué : le numpy.ones empeche le passage du relu. A repenser
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        grad = dtorch.jtensors.JTensors(-1 * (np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[0].shape, unpacked_tensors[1].shape)).reshape(*unpacked_tensors[1].shape)
                                                               if right_size < left_size
                                                               else base_tensor.numpy().reshape(*unpacked_tensors[1].shape)))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def pow_deriv(base_tensor, *tensors):

    tensor : dtorch.jtensors.JTensors = tensors[0]
    fact : int = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += dtorch.jtensors.JTensors(base_tensor() * fact * np.power(tensor(), fact - 1))
        else:
            tensor.grad = dtorch.jtensors.JTensors(base_tensor() * fact * np.power(tensor(), fact - 1))
        tensor.backward(tensor.grad, forced=True)

