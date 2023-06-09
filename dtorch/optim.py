""" imports """
from dtorch.nn import Parameter
import dtorch.jtensors as jtensors
import dtorch.functionnal as fn
import numpy as np
from typing import Tuple
from dtorch.typing import types, DtAny


class OptimizerParam:

    def __init__(self, params : list[Parameter], lr : float = 1e-3, momentum : float = 0.0) -> None:
        
        self.params : list[Parameter] = params
        self.lr : float = lr
        self.momentum : float = momentum
        self.velocity : list[np.ndarray] = [fn.zeros_like(param.get()) for param in self.params]


class Optimizer:

    """ private """

    def __apply__(self) -> None:
        pass

    """ public """

    def zero_grad(self):
        pass


    def step(self, *args):
        pass


class SGD(Optimizer):

    """ private """

    @types(self = DtAny, params = list, lr=float, momentum=float, return_type=None)
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


class AdamParam:

    @types(self = DtAny, params = list[Parameter], lr = float, betas = Tuple[float, float], eps = float, weight_decay = float, return_type = None)
    def __init__(self,
                 params : list[Parameter],
                 lr : float = 0.001,
                 betas : Tuple[float, float] = (0.9, 0.999),
                 eps : float = 1e-08,
                 weight_decay : float = 0.0) -> None:
        
        assert (isinstance(lr, float)), "Invalid learning rate given as argument"
        assert (isinstance(betas, Tuple)), "Invalid betas given as argument"
        assert (isinstance(eps, float)), "Invalid eps given as argument"
        assert (isinstance(weight_decay, float)), "Invalid weight_decay given as argument"

        self.params : list[Parameter] = params
        self.lr : float = lr
        self.betas : Tuple[float, float] = betas
        self.moment : list[np.ndarray] = [fn.zeros_like(param.get()) for param in params]
        self.second_moment : list[np.ndarray] = [fn.zeros_like(param.get()) for param in params]
        self.eps : float = eps
        self.weight_decay : float = weight_decay
        self.t : int = 0
    

class Adam(Optimizer):

    """ Private """

    def __init__(self,
                 params : list[Parameter] | list[AdamParam],
                 lr : float = 0.001,
                 betas : Tuple[float, float] = (0.9, 0.999),
                 eps : float = 1e-08,
                 weight_decay : float = 0.0) -> None:
        
        assert (len(params) > 0), "List of parameters can't be empty"
        assert (isinstance(lr, float)), "Invalid learning rate given as argument"
        assert (isinstance(betas, Tuple)), "Invalid betas given as argument"
        assert (isinstance(eps, float)), "Invalid eps given as argument"
        assert (isinstance(weight_decay, float)), "Invalid weight_decay given as argument"

        if (isinstance(params[0], Parameter)):
            self.__params : list[Parameter] = params
            self.__lr : float = lr
            self.__betas : Tuple[float, float] = betas
            self.__eps : float = eps
            self.__weight_decay : float = weight_decay
            self.__moment : list[np.ndarray] = [fn.zeros_like(param.get()) for param in self.__params]
            self.__second_moment : list[np.ndarray] = [fn.zeros_like(param.get()) for param in self.__params]
            self.__t : int = 0
            self.__s_param = True
        else:
            self.__param_list : list[AdamParam] = params
            self.__s_param = False


    def __apply__(self,
                  tensor : jtensors.JTensors,
                  lr : float,
                  moment : jtensors.JTensors,
                  second_moment : jtensors.JTensors,
                  betas : Tuple[float, float],
                  eps : float,
                  weight_decay : float,
                  t : int) -> None:
        
        if (tensor.grad is None):
            return
        
        if (weight_decay != 0):
            tensor.grad = tensor.grad + weight_decay * tensor.detach()

        moment.update(betas[0] * moment.numpy() + (1 - betas[0]) * tensor.grad.numpy())
        second_moment.update(betas[1] * second_moment.numpy() + (1 - betas[1]) * (tensor.grad.numpy() ** 2))
        m1 = moment.numpy() / (1 - (betas[0] ** t))
        m2 = second_moment.numpy() / (1 - (betas[1] ** t))

        tensor.update(tensor.numpy() - ((lr * m1) / (np.sqrt(m2) + eps)))


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
            self.__t += 1
            for i in range(len(self.__params)):
                self.__apply__(self.__params[i].get(),
                               self.__lr,
                               self.__moment[i],
                               self.__second_moment[i],
                               self.__betas,
                               self.__eps,
                               self.__weight_decay,
                               self.__t)
        else:
            for params in self.__param_list:
                params.t += 1
                for i in range(len(params.params)):
                    self.__apply__(params.params[i].get(),
                                   params.lr,
                                   params.moment[i],
                                   params.second_moment[i],
                                   params.betas,
                                   params.eps,
                                   params.weight_decay,
                                   params.t)

