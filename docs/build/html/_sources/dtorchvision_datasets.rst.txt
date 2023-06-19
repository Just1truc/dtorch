dtorchvision.datasets
=====================

Description
-----------

This package contains some datasets for Image computation.
Here's and example of how to use it:

.. code:: python

    from dtorchvision.datasets import MNISTDataset

    dataset = MNISTDataset(download=True)

    (x, _), _ = dataset.data


Datasets
--------

.. py:class:: MNISTDataset(JDataset)

    The MNIST dataset.

    .. py:method:: __init__(root: str = './data', split: Tuple[str] | str = ('train', 'test'), download: bool = False)

        :param root: The root directory where the dataset will be stored.
        :param split: The split to be used. Can be a tuple of strings or a single string.
        :param download: If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    .. py:attribute:: data
        :type: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

        The data of the dataset. It is a tuple of two tuples, the first one containing the training data and the second one containing the test data. Each tuple contains two tensors, the first one containing the images and the second one containing the labels.

    .. py:method:: train_data()

        :return: The training data of the dataset. (x, y)

    .. py:method:: test_data()

        :return: The test data of the dataset. (x, y)

    
