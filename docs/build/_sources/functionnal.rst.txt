dtorch.functionnal
==================

Description
-----------

This module contain all the methods that can be used on jtensors.
All of them support the autograd and are often used in ML.


Functions
---------

.. py:function:: transpose(tensor : dtorch.jtensors.JTensors, axis : Optional[Tuple[int, int]] = (1, 0)) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to transpose
    :param axis: the axis to transpose
    :return: the transposed tensor

    This function transpose the tensor along the given axes.

.. py:function:: split(tensor : dtorch.jtensors.JTensors, value : int | list[int]) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to split
    :param value: the value to split the tensor
    :return: the splitted tensor

    This function split the tensor along the given axis.


.. py:function:: norm(tensor) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to normalize
    :return: the normalized tensor

    This function return a 1 element tensor containing the norm of the tensor.

.. py:function:: sqrt(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to sqrt
    :return: the sqrted tensor

    This function return a tensor containing the square root of the tensor.

.. py:function:: logsumexp(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to logsumexp
    :return: the logsumexped tensor

    This function return a tensor containing the logsumexp of the tensor.
    Mathematically, it is defined as :math:`log(sum(exp(tensor)))`.

.. py:function:: sigmoid(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors
    
    :param tensor: the tensor to sigmoid
    :return: the sigmoided tensor

    This function return a tensor containing the sigmoid of the tensor.
    Mathematically, it is defined as :math:`1 / (1 + exp(-tensor))`.

.. py:function:: from_list(data : list[dtorch.jtensors.JTensors]) -> dtorch.jtensors.JTensors

    :param data: the list of tensors to concatenate
    :return: the concatenated tensor

    This function return a tensor containing the concatenation of the given tensors.

.. py:function:: to_list(tensor : dtorch.jtensors.JTensors) -> list[dtorch.jtensors.JTensors]

    :param tensor: the tensor to split
    :return: the splitted tensor

    This function return a list of tensors containing the split of the given tensor.

.. py:function:: as_strided(tensor : dtorch.jtensors.JTensors, shape : Tuple[int, int], strides : Tuple[int, int]) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to transform
    :param shape: the shape of the new tensor
    :param strides: the strides of the new tensor
    :return: the strided tensor

    This function return a tensor with the new stride and shape.
    To know more about strides, see `this <https://stackoverflow.com/questions/16798888/what-is-a-stride-in-pandas>`_.

.. py:function:: conv1d(input : dtorch.jtensors.JTensors, weight : dtorch.jtensors.JTensors, bias : Optional[dtorch.jtensors.JTensors] = None, stride : Optional[int] = 1) -> dtorch.jtensors.JTensors

    :param input: 1d tensor (batch_size, in_channel, width) or (in_channel, width)
    :param weight: (out_channel, in_channel, kernel_width) or (in_channel, kernel_width)
    :param bias: (out_channel). Defaults to None.
    :param stride: *movement speed* of the kernel. Defaults to 1.
    :return: the convoluted tensor

    This function return a tensor containing the convolution of the input tensor with the weight tensor.
    If the bias is not None, it will be added to the result.

.. py:function:: conv2d(input : dtorch.jtensors.JTensors, weight : dtorch.jtensors.JTensors, bias : Optional[dtorch.jtensors.JTensors] = None, stride : int = 1) -> dtorch.jtensors.JTensors

    :param input: 2d tensor (batch_size, in_channel, height, width) or (in_channel, height, width)
    :param weight: (out_channel, in_channel, kernel_height, kernel_width) or (in_channel, kernel_height, kernel_width)
    :param bias: (out_channel). Defaults to None.
    :param stride: *movement speed* of the kernel. Defaults to 1.
    :return: the convoluted tensor

    This function return a tensor containing the convolution of the input tensor with the weight tensor.
    If the bias is not None, it will be added to the result.

.. py:function:: dropout(tensor : dtorch.jtensors.JTensors, p : float = 0.5)

    :param tensor: the tensor to dropout
    :param p: the probability of dropout. Defaults to 0.5.
    :return: the dropouted tensor

    This function return a tensor containing the dropouted tensor.
    The dropout is a technique used to prevent overfitting.
    It randomly set some values to 0.

.. py::function:: tanh(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to tanh
    :return: the tanhed tensor

    This function return a tensor containing the tanh of the tensor.
    Mathematically, it is defined as :math:`(exp(tensor) - exp(-tensor)) / (exp(tensor) + exp(-tensor))`.

    Here's a graph of the tanh function:

    .. image:: tanh_plot.png
        :width: 400
        :align: center

.. py:function:: softmax(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to softmax
    :return: the softmaxed tensor

    This function return a tensor containing the softmax of the tensor.
    Mathematically, it is defined as :math:`exp(tensor) / sum(exp(tensor))`.

.. py:function:: max(tensor : dtorch.jtensors.JTensors, value : int | float) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to max
    :param value: the value to max the tensor
    :return: the maxed tensor

    This function return a tensor containing the max between the tensor and each value.

.. py:function:: exp(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to exp
    :return: the exped tensor

    This function return a tensor containing the exp of the tensor.

.. py:function:: log(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to log
    :return: the loged tensor

    This function return a tensor containing the log of the tensor.

.. py:function:: matmul(left : dtorch.jtensors.JTensors, right : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param left: the left tensor
    :param right: the right tensor
    :return: the multiplied tensor

    This function return a tensor containing the multiplication of the left tensor with the right tensor.

.. py:function:: sum(tensor : dtorch.jtensors.JTensors, axis : Optional[Tuple[int]] = None, keepdims : bool = False) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to sum
    :param axis: the axis to sum. Defaults to None.
    :param keepdims: whether to keep the dimensions. Defaults to False.
    :return: the summed tensor

    This function return a tensor containing the sum of the tensor.
    If the axis is not None, it will sum the tensor along the given axis.
    If the keepdims is True, it will keep the dimensions of the tensor.

.. py:function:: squeeze(tensor : dtorch.jtensors.JTensors, axis : int) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to squeeze
    :param axis: the axis to squeeze
    :return: the squeezed tensor

    This function return a tensor containing the squeezed tensor.
    It will remove the dimension of the tensor along the given axis.

.. py:function:: unsqueeze(tensor : dtorch.jtensors.JTensors, axis : int) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to unsqueeze
    :param axis: the axis to unsqueeze
    :return: the unsqueezed tensor

    This function return a tensor containing the unsqueezed tensor.
    It will add a dimension of the tensor along the given axis.

.. py:function:: reshape(tensor : dtorch.jtensors.JTensors, shape : Tuple[int]) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to reshape
    :param shape: the new shape of the tensor
    :return: the reshaped tensor

    This function return a tensor containing the reshaped tensor.
    It will reshape the tensor to the given shape.

.. py:function:: arange(start : int, end : int, step : int = 1) -> dtorch.jtensors.JTensors

    :param start: the start of the range
    :param end: the end of the range
    :param step: the step of the range. Defaults to 1.
    :return: the ranged tensor

    This function return a tensor containing the ranged tensor.
    It will create a tensor containing the values from start to end with the given step.

.. py:function:: tensor(list : list | np.ndarray, require_grads : bool = False, dtype : type | np.dtype = np.float64) -> dtorch.jtensors.JTensors

    :param list: the list to convert to tensor
    :param require_grads: whether the tensor require gradients. Defaults to False.
    :param dtype: the type of the tensor. Defaults to np.float64.
    :return: the converted tensor

    This function return a tensor containing the converted tensor.
    It will convert the list to a tensor.

.. py:function:: random(*shape : int) -> dtorch.jtensors.JTensors

    :param shape: the shape of the random tensor
    :return: the random tensor

    This function return a tensor containing the random tensor.
    It will create a tensor containing random values.

.. py:function:: ones(*shape : int, requires_grad : bool = False) -> dtorch.jtensors.JTensors

    :param shape: the shape of the ones tensor
    :param requires_grad: whether the tensor require gradients. Defaults to False.
    :return: the ones tensor

    This function return a tensor containing ones of the given shape.

.. py:function:: zeros(*shape : int, requires_grad : bool = False) -> dtorch.jtensors.JTensors

    :param shape: the shape of the zeros tensor
    :param requires_grad: whether the tensor require gradients. Defaults to False.
    :return: the zeros tensor

    This function return a tensor containing zeros of the given shape.

.. py:function:: xavier(nb_feat : int, size : int, require_grads : Optional[bool] = False) -> dtorch.jtensors.JTensors

    :param nb_feat: the number of features
    :param size: the size of the tensor
    :param require_grads: whether the tensor require gradients. Defaults to False.
    :return: the xavier tensor

    It will create a tensor containing random values following the xavier initialization.

.. py:function:: zeros_like(tensor : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors

    :param tensor: the tensor to zeros like
    :return: the zeros like tensor

    This function return a tensor containing zeros like the given tensor.

.. py:function:: uniform_(_from : float | int, _to : float | int, size : int, require_grads : bool = False)

    :param _from: the start of the range
    :param _to: the end of the range
    :param size: the size of the tensor
    :param require_grads: whether the tensor require gradients. Defaults to False.
    :return: the uniform tensor

    It will create a tensor containing random values following the uniform initialization.
