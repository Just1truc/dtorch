dtorch.nn
=========

Description
-----------

This module contain the layers that can be used to construct models.
Here is an example of a Linear Neural Network:

.. code-block:: python

    from dtorch import nn
    from dtorch.jtensors import JTensors

    class MyModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()

            self.linear_layer1 = nn.Linear(1, 32)
            self.linear_layer2 = nn.Linear(32, 4)


        def forward(self, x : JTensors):
            return self.linear_layer2(self.linear_layer1(x))


    # data
    x = JTensors([2, 3, 4, 5])

    # model
    model = MyModel()

    # call the model
    # give a tensor of size (4,) as it's the output of the layer2
    result = model(x)


Layers
------

.. py:class:: Module

   The module that other layers and modules are based on.

   .. py:method:: save(path : str)

      :param str path: The path where the model will be saved.

      Save the model to a file.

   .. py:method:: load(path : str)

      :param str path: The path of the model.

      Load the model from a file.

   .. py:method:: parameters() -> list[Parameter]

      Return the list of parameters of each submodules contained in the model reccursivly. It's useful to give the control to an *optimizer*. 
    

   .. py:method:: eval()

      Set the dropout of the module and submodule to ``False`` and the ``require_grads`` attribute of the parameters to ``False``.
      This permit a faster forward has gradients material don't need to be kept.


   .. py:method:: train()

      On the contrary of the ``eval`` method, set all dropout and require_grads of parameters to ``True``.


.. py:class:: Linear(Module)

    The most basic layer for applying a linear transformation to the data.

   .. py:method:: __init__(in_features, out_features, bias=True, device=None, dtype=None)
        
        Applies a linear transformation to the incoming data :math:`y=xA^T+b`

        :param int in_features: size of each input sample

        :param int out_features: size of each output sample

        :param bool bias: If set to False, the layer will not learn an additive bias. Default: True

    Example::

        >> m = nn.Linear(20, 30)
        >> input = dtorch.random(128, 20)
        >> output = m(input)
        >> print(output.size())
        (128, 30)


.. py:class:: Sequential(Module)

    A module that when applied, apply each modules that it registered iterativly

   .. py:method:: __init__(self, *modules : Module)
      
      :param Module modules: The list of module to apply when appplying the module.
    
    Ex::

        model = nn.Sequential(
          nn.Linear(1,20),
          nn.ReLU()
        )

        # here is the equivalent of applying a linear then relu layer
        model(x)


.. py:class:: Softmax(Module)

    The equivalent of applying the ``functionnal.softmax`` method.

    Mathematicaly: :math:`softmax(z)_j = \frac{e^{z_j}}{\sum{k=1}^Ke^{z_k}}`

    It's a distribution function used to set result to a pourcentage, usualy used for classification purposes.


.. py:class:: Sigmoid(Module)

   The equivalent of applying ``functionnal.sigmoid`` function.

   Mathematicaly: :math:`f(x) = \frac{1}{1 + e^{-x}}`

   Similar to ``tanh`` but the result is in ``[0, 1]`` where the result of tanh is in ``[-1, 1]``

   It's an activation layer.


.. py:class:: ReLU(Module)

    equivalent to ``max(x, 0)`` in python.

    .. image:: relu_plot.png
      :width: 400

    According to the universal approximation theorem, any function can be represented using linear and relu layers one after another.
    It's actually canceling the backpropagation of the gradients on some neurones. 


.. py:class:: Tanh(Module)

    equivalent to ``functionnal.tanh``

    .. image:: tanh_plot.png
      :width: 400

    It define the hyperbolic tangante to the curve.
    It's an activation layer.


.. py:class:: Dropout(Module)

    A layer that randomly make a neurone fail by not passing the data though to next layer with a given probability.

    .. py:method:: __init__(p : float = 0.5)

        :param float p: The probaility of each neurone failing.

    It's useful for data augmentation and it has be proven to be a key component to less variance.


.. py:class:: RNN(Module)

    A layer (uni / multi couche) that is used for building Reccurent Neural Networks.

    .. py:method:: __init__(input : int, hidden : int, num_layers : int = 1, non_linearity : str = 'tanh', bias : bool = True, batch_first : bool = False, dropout : float = 0)

        :param int input: number of feature per elements.
        :param int hidden: size of the context (the second output of the forward)
        :param int num_layers: number of reccurent layers. Multiple layers cover more efficiently complex functions. Default : 1
        :param str non_linearity: The activation function that is used σ. Default: 'tanh'. Possibles 'tanh', 'sigmoid'.
        :param bool batch_first: Default to ``False``. If false, input shape is ``(sequence_size, batch_size, input_size)`` else ``(batch_size, sequence_size, input_size)``
        :param bool bias: Determine if a weight should be added in the calcul or not. Default : False.
        :param float dropout: Parameter that add a layer between layers and context similarly to the ``ReLU`` layer.

    Mathematicaly, if :math:`h_t` is the hidden state at time t, σ is the non_linearity function, Wi is the input weight and Wh is the hidden weight.
    Then if bias is `False` :math:`h_t = W_i * x_t + W_h * h_{t-1}`
    
    Otherwise, with :math:`B_i` being the bias of the input and :math:`B_h` being the bias of the hidden state: :math:`h_t = W_i * x_t + B_i + W_h * h_{t-1} + B_h`

    :math:`H_0` is usualy initialised at 0, whereas other weights are initialised from the ``xavier`` distribution.

    When calling the model, this one will output a tuple ``(y, hidden_state)``


.. py:class:: Conv1d(Module)

    A layer for convolution with weights that adapt through training.

    .. warning::
        This module has not been tested throughfully yet. If any problem seem to resort, do not hesitate to report about it on the github of the project.

    .. py:method:: __init__(in_channels : int, out_channels : int, kernel_size : int, stride : Optional[int] = 1, bias : bool = True)

        :param int in_channels: The number of channels the data has. Ex: In an image with 3 color, the data may have 3 channels.
        :param int out_channels: Number of channel the output data need to have.
        :param int kernel_size: width of the convolution kernel (the window going through the data).
        :param Optional[int] stride: Default to 1. This parameter define the number of step the window take in space. (Not tested yet.)
        :param bool bias: Define if a bias weight should be used.

    | Input shape: ``(batch, in_channel, width)``
    | Output shape: ``(batch, out_channel, width - kernel_size)`` without counting the stride.

    This layer is often used for a network to understand a complex data like images and audio.


.. py:class:: Conv2d(Module)

    .. note::
        Implemented soon. (within 2 weeks)
    