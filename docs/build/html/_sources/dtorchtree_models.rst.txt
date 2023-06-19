dtorchtree.models
=================

Description
-----------

This package contains pretrained and trainable models for image classification and generation.

Here's an example of how to use the pretrained models.

.. code:: python

    import dtorch
    from dtorchtree.models import IrisDecisionTree

    # Load the pretrained model.
    model = IrisDecisionTree()

    # Predict the class for the input sample.
    X = dtorch.tensor([5.1, 3.5, 1.4, 0.2])
    y = model.predict(X)
    print(y) # "Setosa"


Models
------

.. py:class:: DecisionTree

    Decision tree for classification.

    .. py:method:: __init__(criterion : str = 'gini', max_depth : Optional[int] = None, multiprocesses_threshold : Optional[int] = 0.4)

        :param criterion: The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
        :param max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.
        :param multiprocesses_threshold: The threshold of impurity to use multiprocesses. If the impurity is lower than this threshold, the node is expanded in a single process.

    .. py:method:: fit(X : torch.Tensor, y : torch.Tensor)

        :param X: The training input samples. (n_samples, n_features)
        :param y: The target values (class labels). (n_samples,)

        Build a decision tree classifier from the training set (X, y).

    .. py:method:: predict(X : torch.Tensor) -> torch.Tensor
            
        :param X: The input samples. (n_features,)
        :return: The predicted class.

        Predict the class for the input sample.

    .. py:method:: save(path : str)

        :param path: The path to save the model.

        Save the model to the path.

    .. py:method:: load(path : str)

        :param path: The path to load the model.

        Load the model from the path.


Pretrained Models
-----------------

.. py:class:: IrisDecisionTree(JPretrainedModel)

    .. py:method:: __init__(self, root: str = './models')

        :param root: The root directory of the pretrained model.

        Load the pretrained model.

    .. py:method:: predict(self, X : torch.Tensor) -> torch.Tensor

        :param X: The input samples. (n_samples, n_features)
        :return: The predicted class.

        Predict the class for the input sample.
