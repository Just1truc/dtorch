""" imports """
import dtorch.jtensors as jt
import unittest
import torch as t
import numpy as np

class TestJTensors(unittest.TestCase):

    def test_jtensor_creation(self):
        """Test creation of jtensor
        """

        # test creation of jtensor
        tensor = jt.JTensors([1, 2, 3, 4])
        self.assertEqual(tensor.shape(), (4,))
        self.assertEqual(tensor.require_grads, False)
        self.assertEqual(tensor.grad, None)

        # test creation of jtensor with require_grads
        tensor = jt.JTensors([1, 2, 3, 4], require_grads=True)
        self.assertEqual(tensor.shape(), (4,))
        self.assertEqual(tensor.require_grads, True)
        self.assertEqual(tensor.grad, None)

    
    def test_jtensor_addition(self):
        """Test addition of jtensor
        """

        # test addition of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = jt.JTensors([1, 2, 3, 4])
        tensor3 = tensor1 + tensor2
        self.assertEqual(tensor3.shape(), (4,))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([2, 4, 6, 8])).all(), True)

        # test addition of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = tensor1 + tensor2
        self.assertEqual(tensor3.shape(), (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[2, 4], [6, 8]])).all(), True)

        # test addition of jtensor with shape and dtype
        tensor1 = jt.JTensors([3], require_grads=True)
        tensor2 = jt.JTensors([2])
        tensor3 = tensor1 + tensor2
        self.assertEqual(tensor3.shape(), (1,))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3.backward()

        torch_tensor1 = t.tensor([3], dtype=float, requires_grad=True)
        torch_tensor2 = t.tensor([2])
        torch_tensor3 = torch_tensor1 + torch_tensor2
        torch_tensor3.backward()

        self.assertEqual(tensor3(), torch_tensor3.detach().numpy())

        self.assertEqual(tensor1.grad, torch_tensor1.grad.detach().numpy())
        self.assertEqual(tensor2.grad, None)


    def test_jtensor_substraction(self):

        """Test substraction of jtensor
        """

        # test substraction of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = jt.JTensors([1, 2, 3, 4])
        tensor3 = tensor1 - tensor2
        self.assertEqual(tensor3.shape(), (4,))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([0, 0, 0, 0])).all(), True)

        # test substraction of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = tensor1 - tensor2
        self.assertEqual(tensor3.shape(), (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[0, 0], [0, 0]])).all(), True)

        # test substraction of jtensor with shape and dtype
        tensor1 = jt.JTensors([3], require_grads=True)
        tensor2 = jt.JTensors([2])
        tensor3 = tensor1 - tensor2
        self.assertEqual(tensor3.shape(), (1,))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3.backward()

        torch_tensor1 = t.tensor([3], dtype=float, requires_grad=True)
        torch_tensor2 = t.tensor([2])
        torch_tensor3 = torch_tensor1 - torch_tensor2
        torch_tensor3.backward()

        self.assertEqual(tensor3(), torch_tensor3.detach().numpy())

        self.assertEqual(tensor1.grad, torch_tensor1.grad.detach().numpy())
        self.assertEqual(tensor2.grad, None)


    def test_jtensor_multiplication(self):

        """Test multiplication of jtensor
        """

        # test multiplication of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = jt.JTensors([1, 2, 3, 4])
        tensor3 = tensor1 * tensor2
        self.assertEqual(tensor3.shape(), (4,))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([1, 4, 9, 16])).all(), True)

        # test multiplication of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = tensor1 * tensor2
        self.assertEqual(tensor3.shape(), (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[1, 4], [9, 16]])).all(), True)

        # test multiplication of jtensor with shape and dtype
        tensor1 = jt.JTensors([3], require_grads=True)
        tensor2 = jt.JTensors([2])
        tensor3 = tensor1 * tensor2
        self.assertEqual(tensor3.shape(), (1,))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3.backward()

        torch_tensor1 = t.tensor([3], dtype=float, requires_grad=True)
        torch_tensor2 = t.tensor([2])
        torch_tensor3 = torch_tensor1 * torch_tensor2
        torch_tensor3.backward()

        self.assertEqual(tensor3(), torch_tensor3.detach().numpy())

        self.assertEqual(tensor1.grad, torch_tensor1.grad.detach().numpy())
        self.assertEqual(tensor2.grad, None)


    def test_jtensor_division(self):

        # test division of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor2 = jt.JTensors([1, 2, 3, 4])
        tensor3 = tensor1 / tensor2
        self.assertEqual(tensor3.shape(), (4,))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([1, 1, 1, 1])).all(), True)

        # test division of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor2 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = tensor1 / tensor2
        self.assertEqual(tensor3.shape(), (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[1, 1], [1, 1]])).all(), True)

        # test division of jtensor with shape and dtype
        tensor1 = jt.JTensors([3], require_grads=True)
        tensor2 = jt.JTensors([2])
        tensor3 = tensor1 / tensor2
        self.assertEqual(tensor3.shape(), (1,))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3.backward()

        torch_tensor1 = t.tensor([3], dtype=float, requires_grad=True)
        torch_tensor2 = t.tensor([2])
        torch_tensor3 = torch_tensor1 / torch_tensor2
        torch_tensor3.backward()

        self.assertEqual(tensor3(), torch_tensor3.detach().numpy())

        self.assertEqual(tensor1.grad, torch_tensor1.grad.detach().numpy())
        self.assertEqual(tensor2.grad, None)


    def test_jtensor_power(self):

        """Test power of jtensor
        """

        # test power of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor3 = tensor1 ** 2
        self.assertEqual(tensor3.shape(), (4,))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([1, 4, 9, 16])).all(), True)

        # test power of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = tensor1 ** 2
        self.assertEqual(tensor3.shape(), (2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[1, 4], [9, 16]])).all(), True)

        # test power of jtensor with shape and dtype
        tensor1 = jt.JTensors([3], require_grads=True)
        tensor3 = tensor1 ** 2
        self.assertEqual(tensor3.shape(), (1,))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3.backward()

        torch_tensor1 = t.tensor([3], dtype=float, requires_grad=True)
        torch_tensor3 = torch_tensor1 ** 2
        torch_tensor3.backward()

        self.assertEqual(tensor3(), torch_tensor3.detach().numpy())

        self.assertEqual(tensor1.grad, torch_tensor1.grad.detach().numpy())


    def test_unsqueeze(self):

        """Test unsqueeze of jtensor
        """

        # test unsqueeze of jtensor
        tensor1 = jt.JTensors([1, 2, 3, 4])
        tensor3 = tensor1.unsqueeze(0)
        self.assertEqual(tensor3.shape(), (1, 4))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[1, 2, 3, 4]])).all(), True)

        # test unsqueeze of jtensor with shape
        tensor1 = jt.JTensors([[1, 2], [3, 4]])
        tensor3 = tensor1.unsqueeze(0)
        self.assertEqual(tensor3.shape(), (1, 2, 2))
        self.assertEqual(tensor3.require_grads, False)
        self.assertEqual(tensor3.grad, None)
        self.assertEqual((tensor3() == np.array([[[1, 2], [3, 4]]])).all(), True)

        # test unsqueeze of jtensor with shape and dtype
        torch_tensor1 = t.tensor([3], dtype=float, requires_grad=True)
        torch_tensor3 = torch_tensor1.unsqueeze(0)
        torch_tensor3 = torch_tensor1.squeeze(0)
        torch_tensor3.backward()

        tensor1 = jt.JTensors([3], require_grads=True)
        tensor3 = tensor1.unsqueeze(0)
        self.assertEqual(tensor3.shape(), (1, 1))
        self.assertEqual(tensor3.require_grads, True)
        self.assertEqual(tensor3.grad, None)
        tensor3 = tensor3.squeeze(0)
        tensor3.backward()

        self.assertEqual(tensor3(), torch_tensor3.detach().numpy())

        self.assertEqual(tensor1.grad(), torch_tensor1.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()
