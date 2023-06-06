""" imports """
import unittest
import dtorch.nn as nn
import dtorch.jtensors as jt
import torch as t
import numpy as np
import dtorch.optim as optim
import dtorch.functionnal as fn

class TestJModules(unittest.TestCase):
    
    def testLinear(self):
        """ Linear layer 
        """

        # create a linear layer
        linear = nn.Linear(2, 1)
        linear._Linear__weights.get().update(np.array([[4, 3]]))
        linear._Linear__biais.get().update(np.array([[4]]))

        optimizer = optim.SGD(linear.parameters(), lr=0.01)

        # create a tensor
        x = jt.JTensors([[1, 2]], require_grads=True)

        # forward pass
        y = fn.sum(linear(x))

        # check the shape
        self.assertEqual(y.shape(), (1,))

        # torch version
        linear_torch = t.nn.Linear(2, 1)
        optimizer_torch = t.optim.SGD(linear_torch.parameters(), lr=0.01)
        linear_torch.weight = t.nn.Parameter(t.tensor([[4, 3]], requires_grad=True, dtype=float))
        linear_torch.bias = t.nn.Parameter(t.tensor([[4]], requires_grad=True, dtype=float))

        y_pred : t.Tensor = t.sum(linear_torch(t.tensor([[1, 2]], dtype=float, requires_grad=True)))

        y.backward()
        optimizer.step()
        y_pred.backward()
        optimizer_torch.step()

        self.assertTrue(y(), y_pred.detach().numpy())
        self.assertTrue((linear_torch.weight.grad.detach().numpy() == linear._Linear__weights.get().grad()).all(), True)

    
    def testBatchedLinear(self):
        """ Batched linear
        """

        # create a linear layer
        linear = nn.Linear(2, 1)
        linear._Linear__weights.get().update(np.array([[4, 3]]))
        linear._Linear__biais.get().update(np.array([[4]]))

        optimizer = optim.SGD(linear.parameters(), lr=0.01)

        # create a tensor
        x = jt.JTensors([[1, 2], [5, 8], [7, 3]], require_grads=True)

        # forward pass
        y = fn.sum(linear(x))

        # torch version
        linear_torch = t.nn.Linear(2, 1)
        linear_torch.weight = t.nn.Parameter(t.tensor([[4, 3]], requires_grad=True, dtype=float))
        linear_torch.bias = t.nn.Parameter(t.tensor([[4]], requires_grad=True, dtype=float))
        optimizer_torch = t.optim.SGD(linear_torch.parameters(), lr=0.01)

        y_pred : t.Tensor = t.sum(linear_torch(t.tensor([[1, 2], [5, 8], [7, 3]], dtype=float, requires_grad=True)))

        # check the shape
        self.assertEqual(y.shape(), (1,))

        y.backward()
        optimizer.step()
        y_pred.backward()
        optimizer_torch.step()

        self.assertTrue(y(), y_pred.detach().numpy())
        self.assertTrue((linear_torch.weight.grad.detach().numpy() == linear._Linear__weights.get().grad()).all(), True)
        self.assertTrue((linear_torch.bias.grad.detach().numpy() == linear._Linear__biais.get().grad()).all(), True)
        self.assertTrue((linear_torch.weight.detach().numpy() == linear._Linear__weights.get().numpy()).all(), True)
        self.assertTrue((linear_torch.bias.detach().numpy() == linear._Linear__biais.get().numpy()).all(), True)

        # epoch 2
        x = jt.JTensors([[1, 2], [-6, -7], [2, 8]], require_grads=True)
        y = fn.sum(linear(x))
        y_pred : t.Tensor = t.sum(linear_torch(t.tensor([[1, 2], [-6, -7], [2, 8]], dtype=float, requires_grad=True)))

        optimizer.zero_grad()
        y.backward()
        optimizer.step()
        optimizer_torch.zero_grad()
        y_pred.backward()
        optimizer_torch.step()

        self.assertTrue(y.numpy(), y_pred.detach().numpy())
        self.assertTrue((linear_torch.weight.grad.detach().numpy() == linear._Linear__weights.get().grad()).all(), True)
        self.assertTrue((linear_torch.bias.grad.detach().numpy() == linear._Linear__biais.get().grad()).all(), True)
        self.assertTrue((linear_torch.weight.detach().numpy() == linear._Linear__weights.get().numpy()).all(), True)
        self.assertTrue((linear_torch.bias.detach().numpy() == linear._Linear__biais.get().numpy()).all(), True)


    def testRElu(self):

        """ ReLu activation
        """

        linear = nn.Linear(2, 1)
        linear._Linear__weights.get().update(np.array([[4, 3]]))
        linear._Linear__biais.get().update(np.array([[4]]))

        relu = nn.ReLU()

        optimizer = optim.SGD(linear.parameters(), lr=0.01)

        x = jt.JTensors([[1, 2], [-6, -7], [2, 8]], require_grads=True)

        l = linear(x)
        o = relu(l)
        y = fn.sum(o)
        self.assertEqual(y.shape(), (1,))

        linear_torch = t.nn.Linear(2, 1)
        linear_torch.weight = t.nn.Parameter(t.tensor([[4, 3]], requires_grad=True, dtype=float))
        linear_torch.bias = t.nn.Parameter(t.tensor([[4]], requires_grad=True, dtype=float))

        optimizer_torch = t.optim.SGD(linear_torch.parameters(), lr=0.01)

        y_pred : t.Tensor = t.sum(t.relu(linear_torch(t.tensor([[1, 2], [-6, -7], [2, 8]], dtype=float, requires_grad=True))))

        y.backward()
        optimizer.step()
        y_pred.backward()
        optimizer_torch.step()

        self.assertTrue(y(), y_pred.detach().numpy())
        self.assertTrue((linear_torch.weight.grad.detach().numpy() == linear._Linear__weights.get().grad()).all(), True)
        self.assertTrue((linear_torch.bias.grad.detach().numpy() == linear._Linear__biais.get().grad()).all(), True)
        self.assertTrue((linear_torch.weight.detach().numpy() == linear._Linear__weights.get().numpy()).all(), True)
        self.assertTrue((linear_torch.bias.detach().numpy() == linear._Linear__biais.get().numpy()).all(), True)


if __name__ == "__main__":
    unittest.main()






