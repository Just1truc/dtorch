from dtorchtree.models import DecisionTree
from dtorchtree.datasets import IrisDataset
import numpy as np
# from sklearn import tree
# from sklearn import datasets

# r = tree.DecisionTreeClassifier()

# x, y = datasets.load_iris(return_X_y=True)

# r = r.fit(x, y)
# print(r)

# print(tree.export_text(r))

o = DecisionTree()

# Droite et gauche sont inversés.
dataset = IrisDataset()

x, y = dataset.data

o.fit(x, y)
print(o)

o.save("proutee.jt")