""" imports """
import dtorch.jtensors as jt
import dtorch.functionnal as F
import unittest
import torch as t
import numpy as np

class TestFunctionnal(unittest.TestCase):

    def test_sum(self):

        """Test sum function
        """

        # test sum of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = F.sum(tensor1)
        self.assertEqual(tensor2.shape(), (1,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual(tensor2() == np.array([10]), True)

        # test sum of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = F.sum(tensor1)
        self.assertEqual(tensor2.shape(), (1,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual(tensor2() == np.array([10]), True)

        # test sum of jtensor with shape and dtype
        tensor1 = jt.JTensors([3, 5, 3, 56, 9], require_grads=True)
        tensor2 = F.sum(tensor1)
        self.assertEqual(tensor2.shape(), (1,))
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
        self.assertEqual(tensor2.shape(), (4,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([0, np.log(2), np.log(3), np.log(4)])).all(), True)

        # test log of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = F.log(tensor1)
        self.assertEqual(tensor2.shape(), (2, 2))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([[0, np.log(2)], [np.log(3), np.log(4)]])).all(), True)

        # test log of jtensor with shape and dtype
        tensor1 = jt.JTensors([3, 4, 6, 89, 2], require_grads=True)
        tensor2 = F.sum(F.log(tensor1))
        self.assertEqual(tensor2.shape(), (1,))
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
        self.assertEqual(tensor2.shape(), (4,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([np.exp(1), np.exp(2), np.exp(3), np.exp(4)])).all(), True)

        # test exp of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = F.exp(tensor1)
        self.assertEqual(tensor2.shape(), (2, 2))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([[np.exp(1), np.exp(2)], [np.exp(3), np.exp(4)]])).all(), True)

        # test exp of jtensor with shape and dtype
        tensor1 = jt.JTensors([3, 4, 6, 89, 2], require_grads=True)
        tensor2 = F.sum(F.exp(tensor1))
        self.assertEqual(tensor2.shape(), (1,))
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
        self.assertEqual(tensor1.shape(), (3, 4))
        self.assertEqual(tensor1.require_grads, False)
        self.assertEqual(tensor1.grad, None)
        self.assertEqual((tensor1() == np.ones((3, 4))).all(), True)

        # test ones of jtensor with dtype
        tensor1 : jt.JTensors = F.ones(3, 4, requires_grad=True)
        self.assertEqual(tensor1.shape(), (3, 4))
        self.assertEqual(tensor1.require_grads, True)
        self.assertEqual(tensor1.grad, None)

        # test ones of jtensor with dtype
        tensor1 : jt.JTensors = F.ones(3, 4, requires_grad=True)
        self.assertEqual(tensor1.shape(), (3, 4))
        self.assertEqual(tensor1.require_grads, True)
        self.assertEqual(tensor1.grad, None)


    def test_matmul(self):

        """Test matmul function
        """

        # test matmul of jtensor
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = F.matmul(tensor1, tensor2)
        self.assertEqual(tensor3.shape(), (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.matmul(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))).all(), True)

        # test matmul of jtensor with different shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2, 3], [3, 4, 5]])
        tensor3 = F.matmul(tensor1, tensor2)
        self.assertEqual(tensor3.shape(), (2, 3))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.matmul(np.array([[1, 2], [3, 4]]), np.array([[1, 2, 3], [3, 4, 5]]))).all(), True)

        # test matmul with backward
        tensor1 = jt.JTensors([[1, 2], [3, 4]], require_grads=True)
        tensor2 = jt.JTensors([[1, 2, 4], [7, 3, 4]], require_grads=True)
        tensor3 = F.sum(F.matmul(tensor1, tensor2))
        self.assertEqual(tensor3.shape(), (1,))
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
        self.assertEqual(tensor2.shape(), (4,))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([1, 2, 3, 4])).all(), True)

        # test max of jtensor with shape
        tensor1 = jt.JTensors([[1, -2], [3, 4]])
        tensor2 : jt.JTensors = F.max(tensor1, 0)
        self.assertEqual(tensor2.shape(), (2, 2))
        self.assertEqual(tensor2.require_grads, False)
        self.assertEqual(tensor2.grad, None)
        self.assertEqual((tensor2() == np.array([[1, 0], [3, 4]])).all(), True)

