dtorch.optim
============

Description
-----------

This module contain optimizers for ML.
The role of an optimizer is to apply the gradients calculated with optimization for a quicker training of the model.
The most currently used is the ``Adam`` optimizer.

Here's an example of it's usage

.. code:: python

    from dtorch import optim

    # create it by passing the model parameters to it
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    ...

    # during training
    for input, target in dataset:
        # setting all the gradients in the network to 0 or else it will accumulate
        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        # after calculating the gradient with the .backward method, calling the optimizer to apply the gradients.
        optimizer.step()

The ``lr`` parameter is the learning rate. It like a velocity for a fast the weight should be going torwards the solution.
A high learning rate seems good at first but is usually enclined to perform poorly when the approximation need more precision.
You may think of it as a video game character with a speed of a 1000 trying to go in a specific location.


Optimizers
----------

.. py:class:: Optimizer

    The optimizer base module

    .. py:method:: zero_grad()

        Reinitialize gradients for parameters managed by the optimizer

    .. py:method:: step()

        Apply gradient with optimizations provided


.. py:class:: SGD(Optimizer)

    An implementation of the Stochastic Gradient Descent algorithm update part with momentum if precised.

    .. py:method:: __init__(params : list[Parameter] | list[OptimizerParam], lr = 1e-3, momentum : float = 0.0)

        :param Union[list[Parameter],list[OptimizerParam]] params: the parameters of a model or a list of parameters and its settings.
        :param float lr: The learning rate of the optimizer
        :param float momentum: the amount of velocity that should be used when applyong the weights. Between [0, 1].

    With :math:`p_i` being the parameter i, :math:`g_i` being the gradient of the parameter i.
    
    If there is no velocity parameter:
    :math:`p_i = p_{i-1} - lr * g_{i, t}`

    Otherwise, with *v* being the velocity, *m* being the momentum:

    | :math:`v_t = m * v_{t-1} + (1 - m) * g_{i, t}`
    | :math:`p_{i,t} = p_{i,t-1} - lr * v_t`


.. py:class:: Adam(Optimizer)

    Adaptive Moment Estimation is an algorithm for gradient descent that adapt the learning over time.

    .. py:method:: __init__(params : list[Parameter], lr : float = 0.001, betas : Tuple[float, float] = (0.9, 0.999), eps : float = 1e-08, weight_decay : float = 0.0)

        :param list[Parameter] params: the parameters of a model or a list of parameters and its settings.
        :param float lr: The learning rate of the optimizer
        :param Tuple[float] betas: the momentum multiplier and the learning rate speed of decrease
        :param float eps: Additionnal base reduction data for the second mutliplier.
        :param float weight_decay: Basicly a way to implement a Regularization multiplier (L2 type).

    With :math:`p_i` being the parameter i, :math:`g_i` being the gradient of the parameter i.
    
    If there is a weight_decay (*wd*) parameter:
    :math:`g_i = g_i + wd * p_{i, t}`

    Then, with :math:`m_t, v_t` being the momentum vectors, :math:`B_1, B_2` being the betas multipliers:

    | :math:`m_t = B_1 * m_{t-1} + (1 - B_1) * g_i`
    | :math:`v_t = B_2 * v_{t-1} + (1 - B_2) * g_i^2`
    | :math:`a = m_t / (1 - B_1^t)`
    | :math:`b = v_t / (1 - B_2^t)`
    | :math:`p_{i, t} = p_{i, t-1} - lr * a / (\sqrt{b} + eps)`
