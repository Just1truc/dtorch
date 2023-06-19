dtorchtree.datasets
===================

Description
-----------

This module contains a collection of datasets that can be used with the
:mod:`dtorchtree` module.

The datasets are all simple to load.
Here's an example of loading the :class:`dtorchtree.datasets.IrisDataset` dataset:

.. code-block:: python

    from dtorchtree.datasets import IrisDataset

    dataset = IrisDataset()

    for sample in dataset:
        print(sample)

The datasets are all subclasses of :class:`dtorch.dataset.JDataset`.

Datasets
--------

.. py:class:: IrisDataset(dtorch.dataset.JDataset)

    .. py:method:: __init__(root: str = './data', split: Tuple[str] | str = 'train', download: bool = False)

        :param root: The root directory to store the dataset.
        :param split: The split to use. Can be either ``'train'`` or ``'test'``.
        :param download: Whether to download the dataset if it doesn't exist. Will be ignored if the dataset already exists.

        :raises ValueError: If ``split`` is not ``'train'`` or ``'test'``.

    .. py:attribute:: data
        :type: Tuple[torch.Tensor, torch.Tensor]

        X, y data.
