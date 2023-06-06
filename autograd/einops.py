""" imports """
import autograd.jtensors
import numpy as np

def rearrange_backward(base_tensor, *args):
    tensor : autograd.jtensors.JTensors = args[0]

    indexes : list = args[1]
    old_shape = tuple([base_tensor.shape()[i] for i in indexes])
    old_stride = tuple([base_tensor.numpy().strides[i] for i in indexes])

    if (tensor.require_grads):
        tensor.grad += autograd.jtensors.JTensors(np.lib.stride_tricks.as_strided(base_tensor(), old_shape, old_stride))
        tensor.backward(tensor.grad, forced=True)

""" no composition of axis yet """
def rearrange(tensor: autograd.jtensors.JTensors, pattern : str):

    dims : list = []
    o = 0
    for i in range(len(pattern.split('->')[0])):
        o += 1
        if (pattern[i] == ' '):
            continue
        if (pattern[i] == '...'):
            raise Exception("... not implemented")
        dims.append(pattern[i])

    res_dims : list = []
    if (pattern.find('->') == -1):
        raise Exception("Invalid pattern")

    for i in range(o + 2, len(pattern)):
        if (pattern[i] == ' '):
            continue
        if (pattern[i] == '...'):
            raise Exception("... not implemented")
        res_dims.append(pattern[i])

    if len(dims) != len(res_dims):
        raise Exception("Invalid pattern")
    
    if len(dims) != len(tensor.shape()):
        raise Exception("Tensor and pattern have different dimensions")
    
    old_shape = tensor.shape()
    indexes = [dims.index(i) for i in res_dims]
    shape = tuple([old_shape[i] for i in indexes])
    stride = tuple([tensor.numpy().strides[i] for i in indexes])
    return autograd.jtensors.JTensors(
        np.lib.stride_tricks.as_strided(tensor(), shape = shape, strides = stride),
        require_grads = tensor.require_grads,
        operation=autograd.operations.CrossOperationBackward(
              rearrange_backward,
              "RearrangeBackward",
              tensor, indexes
        ) if tensor.require_grads else None
    )