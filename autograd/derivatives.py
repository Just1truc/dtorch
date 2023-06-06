""" imports """
import autograd.jtensors
import numpy as np
from typing import Tuple


""" Complex """

def unsqueeze_backward(base_tensor, *tensors):
    
    tensor : autograd.jtensors.JTensors = tensors[0]
    dim : int = tensors[1]

    if (tensor.require_grads):
        grad = autograd.jtensors.JTensors(np.sum(base_tensor(), axis=dim, keepdims=1))
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


def squeeze_backward(base_tensor, *tensors):
    
    tensor : autograd.jtensors.JTensors = tensors[0]
    dim : int = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(np.expand_dims(base_tensor(), axis=dim))
        else:
            tensor.grad = autograd.jtensors.JTensors(np.expand_dims(base_tensor(), axis=dim))
        tensor.backward(tensor.grad, forced=True)


def reshape_backward(base_tensor, *tensors):
    
    tensor : autograd.jtensors.JTensors = tensors[0]
    old_shape : Tuple[int] = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(base_tensor().reshape(old_shape))
        else:
            tensor.grad = autograd.jtensors.JTensors(base_tensor().reshape(old_shape))
        tensor.backward(tensor.grad, forced=True)


""" op """

def sqrt_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(base_tensor() * (1 / (2 * np.sqrt(tensor()))))
        else:
            tensor.grad = autograd.jtensors.JTensors(base_tensor() * (1 / tensor()))
        tensor.backward(tensor.grad, forced=True)


def real_max_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        grad = autograd.jtensors.JTensors(
            base_tensor() * np.where(tensor() == tensor().max(), 1, 0)
        )
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


def max_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]
    value : float | int = tensors[1]

    if (tensor.require_grads):
        grad = autograd.jtensors.JTensors(
            np.where(tensor() < value, 0, base_tensor())
        )
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


def transpose_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        grad = autograd.jtensors.JTensors(
            np.transpose(base_tensor())
        )
        if (tensor.isLeaf()):
            tensor.grad += grad
        else:
            tensor.grad = grad
        tensor.backward(tensor.grad, forced=True)


# just a matrix on what parameters depend on
def matmul_deriv(base_tensor, *tensors):
    left : autograd.jtensors.JTensors = tensors[0]
    right : autograd.jtensors.JTensors = tensors[1]

    if (left.require_grads):
        if (left.isLeaf()):
            left.grad += autograd.jtensors.JTensors(np.matmul(base_tensor(), right().T))
        else:
            left.grad = autograd.jtensors.JTensors(np.matmul(base_tensor(), right().T))
        left.backward(left.grad, forced=True)

    if (right.require_grads):
        if (right.isLeaf()):
            right.grad += autograd.jtensors.JTensors(np.matmul(left().T, base_tensor()))
        else:
            right.grad = autograd.jtensors.JTensors(np.matmul(left().T, base_tensor()))
        right.backward(right.grad, forced=True)


def sum_deriv(base_tensor, *tensors):
    from_tensor : autograd.jtensors.JTensors = tensors[0]

    if (from_tensor.require_grads):
        tensor = autograd.jtensors.JTensors(np.lib.stride_tricks.as_strided(np.ones(from_tensor.shape()), shape=from_tensor.shape(), strides=from_tensor.stride()))
        if (from_tensor.isLeaf()):
            from_tensor.grad += autograd.jtensors.JTensors(base_tensor() * tensor())
        else:
            from_tensor.grad = autograd.jtensors.JTensors(base_tensor() * tensor())
        from_tensor.backward(from_tensor.grad, forced=True)


""" Basics """

def exp_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(base_tensor() * np.exp(tensor()))
        else:
            tensor.grad = autograd.jtensors.JTensors(base_tensor() * np.exp(tensor()))
        tensor.backward(tensor.grad, forced=True)


def log_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(base_tensor() *  (1 / (tensor())))
        else:
            tensor.grad = autograd.jtensors.JTensors(base_tensor() *  (1 / (tensor())))
        tensor.backward(tensor.grad, forced=True)


def mul_deriv(base_tensor, *tensors):

    unpacked_tensors : list[autograd.jtensors.JTensors | float | int] = list(tensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], autograd.jtensors.JTensors)

    if (unpacked_tensors[0].require_grads):
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += autograd.jtensors.JTensors(base_tensor() * (unpacked_tensors[1]() if is_tensor else unpacked_tensors[1]))
        else:
            unpacked_tensors[0].grad = autograd.jtensors.JTensors(base_tensor() * (unpacked_tensors[1]() if is_tensor else unpacked_tensors[1]))
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)
    
    if (is_tensor and unpacked_tensors[1].require_grads):
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += autograd.jtensors.JTensors(base_tensor() * unpacked_tensors[0]())
        else:
            unpacked_tensors[1].grad = autograd.jtensors.JTensors(base_tensor() * unpacked_tensors[0]())
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def dimensionToSum(left : Tuple[int], right : Tuple[int]) -> Tuple[int]:

    left = list(left)
    right = list(right)
    r = 0
    l = 0

    while (r < len(right)):
        if (right[-1 - r] == 1):
            r += 1
            continue
        if (left[-1 -l] == 1):
            l += 1
            continue
        if (left[-1 -l] == right[-1 -r]):
            r += 1
            l += 1
            continue
        break

    return tuple([i for i in range(len(left) - l)])


def tanh_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(base_tensor() * (1 - np.tanh(tensor()) ** 2))
        else:
            tensor.grad = autograd.jtensors.JTensors(base_tensor() * (1 - np.tanh(tensor()) ** 2))
        tensor.backward(tensor.grad, forced=True)


def to_list_deriv(base_tensor, *tensors):

    index : int = tensors[0]
    tensor : autograd.jtensors.JTensors = tensors[1]

    if (tensor.require_grads):
        if (tensor.grad is None):
            tensor.grad = autograd.jtensors.JTensors(np.zeros(tensor.shape()))
        if (tensor.isLeaf()):
            tensor.grad[index] += base_tensor()
        else:
            tensor.grad[index] = base_tensor()
        tensor.backward(tensor.grad, forced=True)


def dropout_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]
    mask : autograd.jtensors.JTensors = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(np.where(mask, 0, base_tensor.numpy()))
        else:
            tensor.grad = autograd.jtensors.JTensors(np.where(mask, 0, base_tensor.numpy()))
        tensor.backward(tensor.grad, forced=True)


def from_list_deriv(base_tensor, *tensors):

    tensors : list[autograd.jtensors.JTensors] = list(tensors)

    for i in range(len(tensors) - 1, -1, -1):
        if (tensors[i].require_grads):
            if (tensors[i].isLeaf()):
                tensors[i].grad += autograd.jtensors.JTensors(base_tensor()[i])
            else:
                tensors[i].grad = autograd.jtensors.JTensors(base_tensor()[i])
            tensors[i].backward(tensors[i].grad, forced=True)


def add_deriv(base_tensor, *tensors):

    unpacked_tensors : list[autograd.jtensors.JTensors] = list(tensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], autograd.jtensors.JTensors)
    left_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[0].shape())))
    right_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[1].shape())))

    if (unpacked_tensors[0].require_grads):
        grad = autograd.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[1].shape(), unpacked_tensors[0].shape())).reshape(*unpacked_tensors[0].shape())
                                                                if left_size < right_size
                                                                else base_tensor.numpy().reshape(*unpacked_tensors[0].shape()))
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += grad
        else:
            unpacked_tensors[0].grad = grad
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        grad = autograd.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[0].shape(), unpacked_tensors[1].shape())).reshape(*unpacked_tensors[1].shape())
                                                               if right_size < left_size
                                                               else base_tensor.numpy().reshape(*unpacked_tensors[1].shape()))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def div_deriv(base_tensor, *tensors):

    unpacked_tensors : list[autograd.jtensors.JTensors] = list(tensors)

    zis_tensor : bool = isinstance(unpacked_tensors[0], autograd.jtensors.JTensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], autograd.jtensors.JTensors)

    # prout
    if (zis_tensor and unpacked_tensors[0].require_grads):
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += autograd.jtensors.JTensors(base_tensor() * (1 / (unpacked_tensors[1]() if is_tensor else unpacked_tensors[1])))
        else:
            unpacked_tensors[0].grad = autograd.jtensors.JTensors(base_tensor() * (1 / (unpacked_tensors[1]() if is_tensor else unpacked_tensors[1])))
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        grad = autograd.jtensors.JTensors(-base_tensor() * ((unpacked_tensors[0]() if zis_tensor else unpacked_tensors[0]) / np.power(unpacked_tensors[1](), 2)))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def sub_deriv(base_tensor, *tensors):
    # need to be fixed

    unpacked_tensors : list[autograd.jtensors.JTensors] = list(tensors)
    is_tensor : bool = isinstance(unpacked_tensors[1], autograd.jtensors.JTensors)
    left_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[0].shape())))
    right_size : int = len(list(filter(lambda x : x > 1, unpacked_tensors[1].shape())))

    if (unpacked_tensors[0].require_grads):
        grad = autograd.jtensors.JTensors(np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[1].shape(), unpacked_tensors[0].shape())).reshape(*unpacked_tensors[0].shape())
                                                                if left_size < right_size
                                                                else base_tensor.numpy().reshape(*unpacked_tensors[0].shape()))
        if (unpacked_tensors[0].isLeaf()):
            unpacked_tensors[0].grad += grad
        else:
            unpacked_tensors[0].grad = grad
        # probleme diagnostiqué : le numpy.ones empeche le passage du relu. A repenser
        unpacked_tensors[0].backward(unpacked_tensors[0].grad, forced = True)

    if (is_tensor and unpacked_tensors[1].require_grads):
        grad = autograd.jtensors.JTensors(-1 * (np.sum(base_tensor(), axis=dimensionToSum(unpacked_tensors[0].shape(), unpacked_tensors[1].shape())).reshape(*unpacked_tensors[1].shape())
                                                               if right_size < left_size
                                                               else base_tensor.numpy().reshape(*unpacked_tensors[1].shape())))
        if (unpacked_tensors[1].isLeaf()):
            unpacked_tensors[1].grad += grad
        else:
            unpacked_tensors[1].grad = grad
        unpacked_tensors[1].backward(unpacked_tensors[1].grad, forced = True)


def pow_deriv(base_tensor, *tensors):

    tensor : autograd.jtensors.JTensors = tensors[0]
    fact : int = tensors[1]

    if (tensor.require_grads):
        if (tensor.isLeaf()):
            tensor.grad += autograd.jtensors.JTensors(base_tensor() * fact * np.power(tensor(), fact - 1))
        else:
            tensor.grad = autograd.jtensors.JTensors(base_tensor() * fact * np.power(tensor(), fact - 1))
        tensor.backward(tensor.grad, forced=True)

