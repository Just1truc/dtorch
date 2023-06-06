""" imports """
from autograd.nn import Parameter
import autograd.jtensors as jtensors
import autograd.functionnal as fn
import numpy as np

class OptimizerParam:

    def __init__(self, params : list[Parameter], lr : float = 1e-3, momentum : float = 0.0) -> None:
        
        self.params : list[Parameter] = params
        self.lr : float = lr
        self.momentum : float = momentum
        self.velocity : list[np.ndarray] = [fn.zeros_like(param.get()) for param in self.params]


class Optimizer:

    """ public """

    def zero_grad(self):
        pass


    def step(self, *args):
        pass


class SGD(Optimizer):

    """ private """

    def __init__(self, params : list[Parameter] | list[OptimizerParam], lr = 1e-3, momentum : float = 0.0) -> None:

        assert (len(params) > 0), "List of parameters can't be empty"
        assert (isinstance(lr, float)), "Invalid type for learning rate"
        assert (isinstance(momentum, float)), "Invalid type for momentum"

        if (isinstance(params[0], Parameter)):
            self.__params : list[Parameter] = params
            self.__lr : float = lr
            self.__momentum : float = momentum
            self.__velocity : list[np.ndarray] = [fn.zeros_like(param.get()) for param in self.__params]
            self.__s_param = True
        else:
            self.__param_list : list[OptimizerParam] = params
            self.__s_param = False


    def __apply__(self, tensor: jtensors.JTensors, lr : float, velocity : jtensors.JTensors, momentum : float) -> None:

        if (tensor.grad is not None):
            #grads = np.sum(tensor.grad(), axis=0, keepdims=1) / len(tensor.grad())
            grads = tensor.grad.numpy()
            velocity.update(momentum * velocity() + grads)
            tensor.update(tensor() - lr * velocity.numpy())
    
    """ Public """
    
    def zero_grad(self):

        """reinitialize gradients for parameters managed by optimizer
        """

        if self.__s_param:
            for param in self.__params:
                param.get().grad = None
        else:
            for params in self.__param_list:
                for param in params.params:
                    param.get().grad = None


    def step(self):

        """apply gradient with optimizations provided
        """

        if self.__s_param:
            for i in range(len(self.__params)):
                self.__apply__(self.__params[i].get(), self.__lr, self.__velocity[i], self.__momentum)
        else:
            for params in self.__param_list:
                for i in range(len(params.params)):
                    self.__apply__(params.params[i].get(), params.lr, params.velocity[i], params.momentum)

