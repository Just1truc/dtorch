dtorchvision.models
===================

Description
-----------

This module contains model definitions and pretrained weights for popular models.

Here's an example of how to use a pretrained model Autoencoder:

.. code-block:: python

    from dtorchvision.datasets import MNISTDataset
    import dtorchvision.models
    from matplotlib import pyplot as plt
    import random

    autoencoder = dtorchvision.models.MNISTAutoEncoder_128_32()

    dataset = MNISTDataset(download=True)
    (x, _), _ = dataset.data

    a = autoencoder(x)
    img = random.randint(0, len(a) - 1)
    plt.imshow(a[img].reshape(28, 28))
    plt.show()
    plt.imshow(x[img].reshape(28, 28))
    plt.show()
    exit()

This code snippet will show you the autoencoder's reconstruction of a random image from the MNIST dataset and then the original image.

Models
------

.. py:class:: AutoEncoder(dtorch.nn.Module)

    An autoencoder model.

    .. py:method:: __init__(input_size, hidden_size, dp : float = 0.0)

        :param int input_size: The size of the input layer.
        :param int hidden_size: The size of the hidden layer.
        :param float dp: Defaults to 0.0. The dropout probability.

    For instance, the following model:

    .. code-block:: python

        autoencoder = dtorchvision.models.AutoEncoder(784, 128)
    
    Produces the following architecture:

    .. code-block:: text

        AutoEncoder(
          Sequential(
            (0): Linear(in_features=784, out_features=128, bias=True)
            (1): ReLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): ReLU()
            (4): Linear(in_features=128, out_features=784, bias=True)
            (5): ReLU()
          )
        )

Pretrained Models
-----------------

.. py:class:: MNISTAutoEncoder_128_32(JPretrainedModel)

    An autoencoder model pretrained on the MNIST dataset.

    .. py:method:: __init__(root : str = './models')

        :param str root: Defaults to './models'. The root directory to store the model.
