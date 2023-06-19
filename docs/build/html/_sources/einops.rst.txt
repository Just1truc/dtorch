dtorch.einops
=============

Description
-----------

This module has for goal to replicate einops fluidity for data shape transformation while bringing *autograd* on the einops methods.


Functions
---------

.. py:function:: rearrange(tensor, pattern) -> JTensors

    This method is meant to work like *einops's* rearrange.

    Example::

        >> import dtorch.jtensors as dt
        >> u = dt.tensor([[4, 2, 4], [5, 2, 6]])
        >> dt.rearrange(u, 'ab->ba')
        jtensor([[4 5] [2 2] [4 6]])
