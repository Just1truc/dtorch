from dtorchtree.models import DecisionTree
from dtorchtree.datasets import IrisDataset

o = DecisionTree()
dataset = IrisDataset()

x, y = dataset.data

o.load("proutee.jt")
print(o)
