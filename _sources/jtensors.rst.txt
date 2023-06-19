dtorch.jtensors
===============

.. py:currentmodule::
    jtensors

.. py:class:: JTensors

   The tensor class wrapping a numpy array to handle autograd.

   .. py:attribute:: JTensors.grad
      :type: JTensors

      Way to access the gradients of the tensors after the ``backward()`` method was called on a result of an operation on said tensors.      
   
   .. py:attribute:: JTensors.require_grads
      :type: bool

      If set to ``True``, the gradients of this tensor will be calculated when the ``backward()`` method is called.

   .. py:attribute:: JTensors.ndims
      :type: int

      Number of dimensions this tensor's shape have.

   .. py:attribute:: JTensors.shape
      :type: Tuple[int]

      The shape of the tensor.

   .. py:attribute:: JTensors.dtype
      :type: numpy.ndtype

      The type of the tensor's elements.

   .. py:attribute:: JTensors.T
      :type: JTensors

      the transpose of the current tensor.

   .. py:attribute:: JTensors.itemsize
      :type: int

      Size of each elements in bytes.

   .. py:attribute:: JTensors.size
      :type: int

      Number of elements in the tensor.

   .. py:attribute:: JTensors.stride
      :type: int

      The current stride of the tensor (stride * itemsize).

   .. py:method:: backward(base_tensor, forced = False) -> None

        :param JTensors base_tensor: The base tensor the gradients will be accumulated on top of.
        :param bool forced: Only useful internaly (may be used with caution)

        Backpropagate though the network to calculate the gradients of tensor linked. ``require_grads`` attribute need to be set to True to backpropagate.

   .. py:method:: numpy() -> numpy.ndarray

        Transform the tensor into a numpy() array.

   .. py:method:: transpose() -> JTensors

        Transpose the tensor.

   .. py:method:: reshape(*shape) -> JTensors

        :param int shape: The shape of the new tensor. Ex : ``tensor.reshape(1, 2, 3)``

   .. py:method:: max() -> JTensors

        Return a 1 element tensor containing the highest element of the current tensor.

   .. py:method:: sum(axis : Optional[Tuple[int]] = None, keepdims : bool = False) -> JTensors

        :param Optional[Tuple[int]] axis: The axis that will be summed. If not provided, all axis are summed.
        :param bool keepdims: Keep the dimension that are summed. 

        Return a tensor containing the sum over specific axis or over all axis if not provided.

   .. py:method:: rearrange(pattern : str) -> JTensors

        :param str pattern: The pattern to rearrange

        The method is similar to ``einops`` rearrange method.

        Ex::

            >> import dtorch.jtensors as dt
            >> u = dt.tensor([[4, 2, 4], [5, 2, 6]])
            >> dt.rearrange(u, 'ab->ba')
            jtensor([[4 5] [2 2] [4 6]])

   .. py:method:: shuffle() -> JTensors

        Return a shuffled version of the tensor. **Does not support backpropagration**

   .. py:method:: norm() -> JTensors

        Return a one element tensor containing the norm of the tensor.

   .. py:method:: detach() -> JTensors

        Return a copy of the tensor that is not linked to the ancient autograd graph. Useful for *TBPTT* methods
        
   .. py:method:: clone() -> JTensors

        Return a clone of the tensor.
    
   .. py:method:: unsqueeze(dim : int) -> JTensors

        :param int dim: Add a new dimension of size 1 at dimension index given.

   .. py:method:: squeeze(dim : int) -> JTensors

        :param int dim: The dimension to squeeze. (It must be of size 1)
