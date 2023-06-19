dtorch.loss
===========

Description
-----------

A loss function is generaly a way to establish how much errors a model is doing.
Commons loss function are present in this package.

It's usage is pretty straightforward.
Ex:
.. code::python

    >> loss = dtorch.loss.MSELoss()
    >> input = dtorch.random(3, 5, requires_grad=True)
    >> target = dtorch.random(3, 5)
    >> output = loss(input, target)
    >> output.backward()

Loss
----

.. py:class:: MSELoss

    Mean Squared Error is a function that is the mean of the squared residual between two set of data.

    Mathematicaly: :math:`y = \frac{1}{n} \sum_{i=1}^N(y_i - ŷ_i)^2`
    where ŷ is the wanted result.

.. py:class:: BCELoss

    .. warning::
        It is not tested yet

    BCELoss is a way to evaluate how good a prediction is between two classes.
    It's used for binary classification as such.

    Mathematicaly it can be written as: :math:`-\frac{1}{n}\sum_{i=0}^Ny_i*log(ŷ_i)+(1-y_i)*log(1-ŷ_i)`

    