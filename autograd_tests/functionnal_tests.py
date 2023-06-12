""" imports """
import dtorch.jtensors as jt
import dtorch.functionnal as F
import unittest
import torch as t
import numpy as np
import dtorch as dt

class TestFunctionnal(unittest.TestCase):

    def test_sum(self):

        """Test sum function
        """

        # test sum of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = F.sum(tensor1)
        self.assertEqual(tensor2.shape, (1,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual(tensor2() == np.array([10]), True)

        # test sum of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = F.sum(tensor1)
        self.assertEqual(tensor2.shape, (1,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual(tensor2() == np.array([10]), True)

        # test sum of jtensor with shape and dtype
        tensor1 = jt.JTensors([3, 5, 3, 56, 9], require_grads=True)
        tensor2 = F.sum(tensor1)
        self.assertEqual(tensor2.shape, (1,))
        self.assertEqual(tensor2.require_grads, True)
        self.assertEqual(tensor2.grad, None)
        tensor2.backward()

        torch_tensor1 = t.tensor([3, 5, 3, 56, 9], dtype=float, requires_grad=True)
        torch_tensor2 = torch_tensor1.sum()
        torch_tensor2.backward()

        self.assertEqual((tensor1.grad == torch_tensor1.grad.numpy()).all(), True)


    def test_log(self):

        """Test log function
        """

        # test log of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = F.log(tensor1)
        self.assertEqual(tensor2.shape, (4,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([0, np.log(2), np.log(3), np.log(4)])).all(), True)

        # test log of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = F.log(tensor1)
        self.assertEqual(tensor2.shape, (2, 2))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([[0, np.log(2)], [np.log(3), np.log(4)]])).all(), True)

        # test log of jtensor with shape and dtype
        tensor1 = jt.JTensors([3, 4, 6, 89, 2], require_grads=True)
        tensor2 = F.sum(F.log(tensor1))
        self.assertEqual(tensor2.shape, (1,))
        self.assertEqual(tensor2.require_grads, True)
        self.assertEqual(tensor2.grad, None)
        tensor2.backward()

        torch_tensor1 = t.tensor([3, 4, 6, 89, 2], dtype=float, requires_grad=True)
        torch_tensor2 = t.sum(t.log(torch_tensor1))
        torch_tensor2.backward()

        self.assertEqual((tensor1.grad == torch_tensor1.grad.numpy()).all(), True)


    def test_exp(self):

        """Test exp function
        """

        # test exp of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = F.exp(tensor1)
        self.assertEqual(tensor2.shape, (4,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([np.exp(1), np.exp(2), np.exp(3), np.exp(4)])).all(), True)

        # test exp of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = F.exp(tensor1)
        self.assertEqual(tensor2.shape, (2, 2))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([[np.exp(1), np.exp(2)], [np.exp(3), np.exp(4)]])).all(), True)

        # test exp of jtensor with shape and dtype
        tensor1 = jt.JTensors([3, 4, 6, 89, 2], require_grads=True)
        tensor2 = F.sum(F.exp(tensor1))
        self.assertEqual(tensor2.shape, (1,))
        self.assertEqual(tensor2.require_grads, True)
        self.assertEqual(tensor2.grad, None)
        tensor2.backward()

        torch_tensor1 = t.tensor([3, 4, 6, 89, 2], dtype=float, requires_grad=True)
        torch_tensor2 = t.sum(t.exp(torch_tensor1))
        torch_tensor2.backward()

        self.assertEqual((tensor1.grad == torch_tensor1.grad.numpy()).all(), True)


    def test_ones(self):

        """Test ones function
        """

        # test ones of jtensor
        tensor1 = F.ones(3, 4)
        self.assertEqual(tensor1.shape, (3, 4))
        self.assertEqual(tensor1.require_grads, False)
        self.assertEqual(tensor1.grad, None)
        self.assertEqual((tensor1() == np.ones((3, 4))).all(), True)

        # test ones of jtensor with dtype
        tensor1 : jt.JTensors = F.ones(3, 4, requires_grad=True)
        self.assertEqual(tensor1.shape, (3, 4))
        self.assertEqual(tensor1.require_grads, True)
        self.assertEqual(tensor1.grad, None)

        # test ones of jtensor with dtype
        tensor1 : jt.JTensors = F.ones(3, 4, requires_grad=True)
        self.assertEqual(tensor1.shape, (3, 4))
        self.assertEqual(tensor1.require_grads, True)
        self.assertEqual(tensor1.grad, None)


    def test_matmul(self):

        """Test matmul function
        """

        # test matmul of jtensor
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = F.matmul(tensor1, tensor2)
        self.assertEqual(tensor3.shape, (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.matmul(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))).all(), True)

        # test matmul of jtensor with different shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2, 3], [3, 4, 5]])
        tensor3 = F.matmul(tensor1, tensor2)
        self.assertEqual(tensor3.shape, (2, 3))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.matmul(np.array([[1, 2], [3, 4]]), np.array([[1, 2, 3], [3, 4, 5]]))).all(), True)

        # test matmul with backward
        tensor1 = jt.JTensors([[1, 2], [3, 4]], require_grads=True)
        tensor2 = jt.JTensors([[1, 2, 4], [7, 3, 4]], require_grads=True)
        tensor3 = F.sum(F.matmul(tensor1, tensor2))
        self.assertEqual(tensor3.shape, (1,))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3.backward()

        torch_tensor1 = t.tensor([[1, 2], [3, 4]], dtype=float, requires_grad=True)
        torch_tensor2 = t.tensor([[1, 2, 4], [7, 3, 4]], dtype=float, requires_grad=True)
        torch_tensor3 = t.sum(t.matmul(torch_tensor1, torch_tensor2))
        torch_tensor3.backward()

        self.assertEqual((tensor1.grad() == torch_tensor1.grad.numpy()).all(), True)


    def test_max(self):

        # TODO : remake

        """Test max function
        """

        # test max of jtensor
        tensor1 : jt.JTensors = jt.JTensors([1, 2, 3, 4])
        tensor2 = F.max(tensor1, 0)
        self.assertEqual(tensor2.shape, (4,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([1, 2, 3, 4])).all(), True)

        # test max of jtensor with shape
        tensor1 = jt.JTensors([[1, -2], [3, 4]])
        tensor2 : jt.JTensors = F.max(tensor1, 0)
        self.assertEqual(tensor2.shape, (2, 2))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([[1, 0], [3, 4]])).all(), True)


    def test_transpose(self):

        """Test transpose function
        """
        
        a : jt.JTensors = jt.JTensors([[1, 2, 3], [4, 5, 6]])
        b = F.transpose(a)

        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.require_grads, False)

        a = jt.JTensors([[1, 2, 3], [4, 5, 6]])
        b = F.transpose(a, (1, 0))

        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.require_grads, False)

        a = dt.tensor([[1, 2, 3], [4, 5, 6]], require_grads=True)
        b = dt.sum(dt.transpose(a, (1, 0)))

        torch_a = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=float, requires_grad=True)
        torch_b = t.sum(t.transpose(torch_a, 1, 0))

        b.backward()
        torch_b.backward()

        self.assertEqual((a.grad.numpy() == torch_a.grad.numpy()).all(), True)


    def test_sqrt(self):

        """Test sqrt function
        """

        a = jt.JTensors([[1, 2, 3], [4, 5, 6]], require_grads=True)
        b = F.sum(F.sqrt(a))

        a_torch = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=float, requires_grad=True)
        b_torch = t.sum(t.sqrt(a_torch))

        b.backward()
        b_torch.backward()

        self.assertEqual(np.array_equal(np.round(a.grad.numpy(), 4), np.round(a_torch.grad.numpy(), 4)), True)
    

    def test_square(self):

        """ Test square function
        """

        a = jt.JTensors([[1, 2, 3], [4, 5, 6]], require_grads=True)
        b = F.sum(a ** 3)

        a_torch = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=float, requires_grad=True)
        b_torch = t.sum(a_torch ** 3)

        b.backward()
        b_torch.backward()

        self.assertEqual(np.array_equal(a.grad.numpy(), a_torch.grad.numpy()), True)


    def test_norm(self):

        """ Test norm function
        """

        a : jt.JTensors = dt.tensor([[1, 2, 3], [4, 5, 6]], require_grads=True)

        b = F.norm(a)

        self.assertEqual(b.shape, (1,))

        torch_a = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=float, requires_grad=True)

        torch_b = t.norm(torch_a)

        b.backward()
        torch_b.backward()

        self.assertEqual(round(float(b()), 4) == round(float(torch_b.detach().numpy()), 4), True)
        self.assertEqual((np.round(a.grad.numpy(), 4) == np.round(torch_a.grad.numpy(), 4)).all(), True)

    
    def test_as_strided(self):

        a = dt.tensor([[1, 2, 3], [4, 5, 6]], require_grads=True)
        b = dt.as_strided(a, (2, 2), (1, 1))

        torch_a = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=float, requires_grad=True)
        torch_b = t.as_strided(torch_a, (2, 2), (1, 1))

        self.assertEqual(b.shape, torch_b.shape)
        self.assertEqual((b() == torch_b.detach().numpy()).all(), True)

        b = dt.sum(b * dt.tensor([[1, 2], [3, 4]], require_grads=True))
        torch_b = t.sum(torch_b * t.tensor([[1, 2], [3, 4]], dtype=float, requires_grad=True))

        b.backward()
        torch_b.backward()

        self.assertEqual((b() == torch_b.detach().numpy()).all(), True)

        self.assertEqual((a.grad.numpy() == torch_a.grad.numpy()).all(), True)


    def test_conv1d(self):

        a = dt.tensor([[[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]], require_grads=True)
        b = dt.tensor([[[1, 2, 3], [6, 1, 8]]], require_grads=True)

        c = dt.conv1d(a, b)

        torch_a = t.tensor([[[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]], dtype=float, requires_grad=True)
        torch_b = t.tensor([[[1, 2, 3], [6, 1, 8]]], dtype=float, requires_grad=True)

        torch_c = t.conv1d(torch_a, torch_b)

        self.assertEqual(c.shape, torch_c.shape)
        self.assertEqual(np.array_equal(c(), torch_c.detach().numpy()), True)

        c = dt.sum(c * 4)

        torch_c = t.sum(torch_c * 4)

        c.backward()
        torch_c.backward()

        self.assertEqual(np.array_equal(c[0], torch_c.item()), True)
        self.assertEqual(np.array_equal(a.grad.numpy(), torch_a.grad.numpy()), True)


    def test_conv1d_batched(self):

        # batched test

        a = dt.tensor([[[1, 2, 3, 4, 5, 6]], [[10, 11, 12, 13, 14, 15]]], require_grads=True)

        b = dt.tensor([[[1, 4]]], require_grads=True)

        c = dt.conv1d(a, b)

        torch_a = t.tensor([[[1, 2, 3, 4, 5, 6]], [[10, 11, 12, 13, 14, 15]]], dtype=float, requires_grad=True)
        torch_b = t.tensor([[[1, 4]]], dtype=float, requires_grad=True)
        torch_c = t.conv1d(torch_a, torch_b)

        #print(c, torch_c)
        self.assertEqual(c.shape, torch_c.shape)
        self.assertEqual(np.array_equal(c(), torch_c.detach().numpy()), True)

        c = dt.sum(c * 4)

        torch_c = t.sum(torch_c * 4)

        c.backward()
        torch_c.backward()

        self.assertEqual(np.array_equal(c[0], torch_c.item()), True)
        self.assertEqual(np.array_equal(a.grad.numpy(), torch_a.grad.numpy()), True)


    def test_conv1d_batched_2(self):

        # batched test 2

        a = dt.tensor([[[1, 2, 3, 4, 5, 6]], [[10, 11, 12, 13, 14, 15]]], require_grads=True)

        b = dt.tensor([[[1, 4]], [[1, 4]], [[1, 4]]], require_grads=True)

        c = dt.conv1d(a, b)

        torch_a = t.tensor([[[1, 2, 3, 4, 5, 6]], [[10, 11, 12, 13, 14, 15]]], dtype=float, requires_grad=True)
        torch_b = t.tensor([[[1, 4]], [[1, 4]], [[1, 4]]], dtype=float, requires_grad=True)
        torch_c = t.conv1d(torch_a, torch_b)

        #print(c, torch_c)
        self.assertEqual(c.shape, torch_c.shape)
        self.assertEqual(np.array_equal(c(), torch_c.detach().numpy()), True)

        c = dt.sum(c * 4)

        torch_c = t.sum(torch_c * 4)

        c.backward()
        torch_c.backward()

        self.assertEqual(np.array_equal(c[0], torch_c.item()), True)
        self.assertEqual(np.array_equal(a.grad.numpy(), torch_a.grad.numpy()), True)


    def test_conv2d(self):

        a = dt.tensor([[[[1, 2], [3, 4], [5, 6]], [[10, 11], [12, 13], [14, 15]]]], require_grads=True)
        b = dt.tensor([[[[1, 4], [3, 6]], [[8, 3], [3, 7]]]], require_grads=True)

        c = dt.conv2d(a, b)

        torch_a = t.tensor([[[[1, 2], [3, 4], [5, 6]], [[10, 11], [12, 13], [14, 15]]]], dtype=float, requires_grad=True)
        torch_b = t.tensor([[[[1, 4], [3, 6]], [[8, 3], [3, 7]]]], dtype=float, requires_grad=True)

        torch_c = t.conv2d(torch_a, torch_b)

        self.assertEqual(c.shape, torch_c.shape)
        self.assertEqual(np.array_equal(c(), torch_c.detach().numpy()), True)

        c = dt.sum(c * 4)

        torch_c = t.sum(torch_c * 4)

        c.backward()
        torch_c.backward()

        self.assertEqual(np.array_equal(c[0], torch_c.item()), True)
        self.assertEqual(np.array_equal(a.grad.numpy(), torch_a.grad.numpy()), True)

        # batched tests

        a = dt.tensor([[[[1, 2], [3, 4], [5, 6]], [[10, 11], [12, 13], [14, 15]]], [[[21, 22], [23, 24], [25, 26]], [[30, 31], [32, 33], [34, 35]]]], require_grads=True)
        b = dt.tensor([[[[1, 4], [3, 6]], [[8, 3], [3, 7]]], [[[1, 4], [3, 6]], [[8, 3], [3, 7]]]], require_grads=True)

        c = dt.conv2d(a, b)

        torch_a = t.tensor([[[[1, 2], [3, 4], [5, 6]], [[10, 11], [12, 13], [14, 15]]], [[[21, 22], [23, 24], [25, 26]], [[30, 31], [32, 33], [34, 35]]]], dtype=float, requires_grad=True)
        torch_b = t.tensor([[[[1, 4], [3, 6]], [[8, 3], [3, 7]]], [[[1, 4], [3, 6]], [[8, 3], [3, 7]]]], dtype=float, requires_grad=True)

        torch_c = t.conv2d(torch_a, torch_b)

        #print(c, torch_c)

        self.assertEqual(c.shape, torch_c.shape)
        self.assertEqual(np.array_equal(c(), torch_c.detach().numpy()), True)

        c = dt.sum(c * 4)

        torch_c = t.sum(torch_c * 4)

        c.backward()
        torch_c.backward()

        self.assertEqual(np.array_equal(c[0], torch_c.item()), True)
        self.assertEqual(np.array_equal(a.grad.numpy(), torch_a.grad.numpy()), True)
