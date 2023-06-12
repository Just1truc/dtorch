""" imports """
import unittest
import dtorch.nn as nn
import dtorch.jtensors as jt
import torch as t
import numpy as np
import dtorch.optim as optim
import dtorch.functionnal as fn
import dtorch.loss as ls
import dtorch as dt

class MyBalls(nn.Module):

    def __init__(self):
    
        super().__init__()
        self.lstm = nn.RNN(1, 1, 2, batch_first=True, bias=True)
    
    def forward(self, x):
        res, h = self.lstm(x)
        return res, h

class MyTorchBalls(t.nn.Module):
    
    def __init__(self):
        super(MyTorchBalls, self).__init__()
        self.rnn = t.nn.RNN(1, 1, 2, batch_first=True, bias=True)

    def forward(self, x):
        return self.rnn(x)
    

class MyConv1d(nn.Module):

    def __init__(self):
    
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 2, bias=True)
    
    def forward(self, x):
        return self.conv(x)
    

class MyTorchConv1d(t.nn.Module):

    def __init__(self):
    
        super().__init__()
        self.conv = t.nn.Conv1d(1, 1, 2, bias=True)
    
    def forward(self, x):
        return self.conv(x)


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
        self.assertEqual(y.shape, (1,))

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
        self.assertEqual(y.shape, (1,))

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
        self.assertEqual(y.shape, (1,))

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


    def testRNN(self):
            
        torch_model = MyTorchBalls()
        model = MyBalls()
        torch_model.rnn.weight_hh_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.weight_ih_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_hh_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_ih_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.weight_hh_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.weight_ih_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_hh_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_ih_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        model.lstm.weight_hh_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.weight_ih_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.bias_hh_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.bias_ih_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.weight_hh_l0 = nn.Parameter(jt.JTensors([[0.5]]), 'weight_hh_l0')
        model.lstm.weight_ih_l0 = nn.Parameter(jt.JTensors([[0.5]]), 'weight_ih_l0')
        model.lstm.bias_hh_l0 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.bias_ih_l0 = nn.Parameter(jt.JTensors([[0.5]]))

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer_torch = t.optim.SGD(torch_model.parameters(), lr=0.01)

        x_train_tensor = jt.JTensors([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
        y_train_tensor = jt.JTensors([[[2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]])
        x_torch = t.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
        y_torch = t.tensor([[[2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]])

        loss = ls.MSELoss()
        loss_torch = t.nn.MSELoss()

        optimizer.zero_grad()
        m, prout = model.forward(x_train_tensor)
        optimizer_torch.zero_grad()
        m_torch, prout_torch = torch_model.forward(x_torch)
        a = loss(m, y_train_tensor)
        a.backward()
        a = model.lstm.weight_hh_l0.get().grad
        b = model.lstm.weight_ih_l0.get().grad
        optimizer.step()
        u_torch = loss_torch(m_torch, y_torch)
        u_torch.backward()
        c = torch_model.rnn.weight_hh_l0.grad
        d = torch_model.rnn.weight_ih_l0.grad
        optimizer_torch.step()

        self.assertTrue((round(float(a.detach().numpy()), 4) == round(float(c.detach().numpy()), 4)), True)
        self.assertTrue((round(float(b.detach().numpy()), 4) == round(float(d.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.weight_hh_l0.get().numpy()), 4) == round(float(torch_model.rnn.weight_hh_l0.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.weight_ih_l0.get().numpy()), 4) == round(float(torch_model.rnn.weight_ih_l0.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.bias_hh_l0.get().numpy()), 4) == round(float(torch_model.rnn.bias_hh_l0.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.bias_ih_l0.get().numpy()), 4) == round(float(torch_model.rnn.bias_ih_l0.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.weight_hh_l1.get().numpy()), 4) == round(float(torch_model.rnn.weight_hh_l1.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.weight_ih_l1.get().numpy()), 4) == round(float(torch_model.rnn.weight_ih_l1.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.bias_hh_l1.get().numpy()), 4) == round(float(torch_model.rnn.bias_hh_l1.detach().numpy()), 4)), True)
        self.assertTrue((round(float(model.lstm.bias_ih_l1.get().numpy()), 4) == round(float(torch_model.rnn.bias_ih_l1.detach().numpy()), 4)), True)

        optimizer_torch.step()


    def testAdam(self):

        torch_model = MyTorchBalls()
        model = MyBalls()
        torch_model.rnn.weight_hh_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.weight_ih_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_hh_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_ih_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.weight_hh_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.weight_ih_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_hh_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        torch_model.rnn.bias_ih_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
        model.lstm.weight_hh_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.weight_ih_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.bias_hh_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.bias_ih_l1 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.weight_hh_l0 = nn.Parameter(jt.JTensors([[0.5]]), 'weight_hh_l0')
        model.lstm.weight_ih_l0 = nn.Parameter(jt.JTensors([[0.5]]), 'weight_ih_l0')
        model.lstm.bias_hh_l0 = nn.Parameter(jt.JTensors([[0.5]]))
        model.lstm.bias_ih_l0 = nn.Parameter(jt.JTensors([[0.5]]))

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer_torch = t.optim.Adam(torch_model.parameters(), lr=0.01)

        x_train_tensor = jt.JTensors([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
        y_train_tensor = jt.JTensors([[[2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]])

        x_torch = t.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
        y_torch = t.tensor([[[2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]])

        optimizer.zero_grad()
        optimizer_torch.zero_grad()

        loss = ls.MSELoss()
        torch_loss = t.nn.MSELoss()

        i, _ = model.forward(x_train_tensor)
        j, _ = torch_model(x_torch)
        m = loss(i, y_train_tensor)
        m_torch = torch_loss(j, y_torch)

        self.assertEqual(round(float(m.detach().numpy()), 4), round(float(m_torch.detach().numpy()), 4))

        m.backward()
        m_torch.backward()


        self.assertEqual(round(float(model.lstm.weight_hh_l1.get().grad.numpy()), 4), round(float(torch_model.rnn.weight_hh_l1.grad.numpy()), 4))
        self.assertEqual(round(float(model.lstm.weight_ih_l1.get().grad.numpy()), 4), round(float(torch_model.rnn.weight_ih_l1.grad.numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_hh_l1.get().grad.numpy()), 4), round(float(torch_model.rnn.bias_hh_l1.grad.numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_ih_l1.get().grad.numpy()), 4), round(float(torch_model.rnn.bias_ih_l1.grad.numpy()), 4))
        self.assertEqual(round(float(model.lstm.weight_hh_l0.get().grad.numpy()), 4), round(float(torch_model.rnn.weight_hh_l0.grad.numpy()), 4))
        self.assertEqual(round(float(model.lstm.weight_ih_l0.get().grad.numpy()), 4), round(float(torch_model.rnn.weight_ih_l0.grad.numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_hh_l0.get().grad.numpy()), 4), round(float(torch_model.rnn.bias_hh_l0.grad.numpy()), 4))

        optimizer.step()
        optimizer_torch.step()

        self.assertEqual(round(float(model.lstm.weight_hh_l0.get().numpy()), 4), round(float(torch_model.rnn.weight_hh_l0.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.weight_ih_l0.get().numpy()), 4), round(float(torch_model.rnn.weight_ih_l0.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_hh_l0.get().numpy()), 4), round(float(torch_model.rnn.bias_hh_l0.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_ih_l0.get().numpy()), 4), round(float(torch_model.rnn.bias_ih_l0.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.weight_hh_l1.get().numpy()), 4), round(float(torch_model.rnn.weight_hh_l1.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.weight_ih_l1.get().numpy()), 4), round(float(torch_model.rnn.weight_ih_l1.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_hh_l1.get().numpy()), 4), round(float(torch_model.rnn.bias_hh_l1.detach().numpy()), 4))
        self.assertEqual(round(float(model.lstm.bias_ih_l1.get().numpy()), 4), round(float(torch_model.rnn.bias_ih_l1.detach().numpy()), 4))


    def test_conv1d(self):

        model = MyConv1d()
        torch_model = MyTorchConv1d()
        
        model.conv.weights = nn.Parameter(dt.tensor([[[0.5, 0.5, 0.5]]], require_grads=True))
        model.conv.biais = nn.Parameter(dt.tensor([0.5], require_grads=True))

        torch_model.conv.weight = t.nn.Parameter(t.tensor([[[0.5, 0.5, 0.5]]], requires_grad=True, dtype=t.float32))
        torch_model.conv.bias = t.nn.Parameter(t.tensor([0.5], requires_grad=True, dtype=t.float32))

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer_torch = t.optim.Adam(torch_model.parameters(), lr=0.01)

        x_train_tensor = jt.JTensors([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        y_train_tensor = jt.JTensors([[[2.0, 3.0, 4.0, 5.0]]])

        x_torch = t.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        y_torch = t.tensor([[[2.0, 3.0, 4.0, 5.0]]])

        optimizer.zero_grad()
        optimizer_torch.zero_grad()

        loss = ls.MSELoss()
        torch_loss = t.nn.MSELoss()

        i = model(x_train_tensor)
        j = torch_model(x_torch)
        m = loss(i, y_train_tensor)
        m_torch = torch_loss(j, y_torch)

        self.assertEqual(round(float(m.detach().numpy()), 4), round(float(m_torch.detach().numpy()), 4))

        m.backward()
        m_torch.backward()

        self.assertEqual(np.array_equal(model.conv.weights.get().grad.numpy(), torch_model.conv.weight.grad.detach().numpy()), True)
        self.assertEqual(np.array_equal(model.conv.biais.get().grad.numpy(), torch_model.conv.bias.grad.detach().numpy()), True)


if __name__ == "__main__":
    unittest.main()






