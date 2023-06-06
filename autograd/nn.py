""" imports """
from typing import Any
import autograd
import functools
from math import sqrt
import pickle

class Parameter:

    def __init__(self, tensor : autograd.jtensors.JTensors, name : str = "") -> None:
        
        self.__tensor : autograd.jtensors.JTensors = tensor
        self.__tensor.require_grads = True
        self.__tensor.add_name(name)


    def get(self) -> autograd.jtensors.JTensors:
        return self.__tensor
    

    def __str__(self) -> str:
        return "Parameter(" + self.__tensor.__str__() + ")"


    def __repr__(self) -> str:
        return self.__str__() 


class Module:

    def __init__(self) -> None:
        pass

    # transform submodules into dict for serialization
    _serialized_sub_modules : list = []
    _sub_modules : list = []
    _parameters : list[Parameter] = []

    __last = None
    # count self
    _sub : int = 0


    """ private """

    def __setattr__(self, __name: str, __value: Any) -> None:
        
        if (isinstance(__value, Module)):
            Module._sub_modules.append(__value)

        if (isinstance(__value, Parameter)):
            Module._parameters.append(__value)
            if (Module.__last is not self):
                if (Module.__last is not None):
                    Module._sub += 1
                Module.__last = self
                Module._serialized_sub_modules.append({})

            Module._serialized_sub_modules[Module._sub][__name] = __value

        self.__dict__.__setitem__(__name, __value)

    """ public """

    def parameters(self):
        return self._parameters
    

    def eval(self) -> None:
        """set the model in eval mode. Reccursivly set require_grads of parameters to false 
        """

        for parameter in Module._parameters:
            parameter.get().require_grads = False
        for module in Module._sub_modules:
            if (hasattr(self, "_dp_rule")):
                module._dp_rule = False


    def train(self) -> None:
        """Set model in train mode. Require grad for all parameters
        """

        for parameter in Module._parameters:
            parameter.get().require_grads = True
        for module in Module._sub_modules:
            if (hasattr(self, "_dp_rule")):
                module._dp_rule = True

    
    def save(self, path : str) -> None:
        """Save the model in the given path
        """
        pickle.dump(self._serialized_sub_modules, open(path, "wb"))


    def load(self, path : str) -> None:
        """Load the model from the given path
        """
        try:
            self._serialized_sub_modules = pickle.load(open(path, "rb"))
            # clean current model
            Module._parameters = []
            Module._serialized_sub_modules = []
            Module._sub = 0
            Module.__last = None
            # load new model
            for id, sub_module in enumerate(self._serialized_sub_modules):
                for name, param in sub_module.items():
                    Module._sub_modules[id].__setattr__(name, param)
        except:
            raise Exception("Save file is not compatible with current model")

class Linear(Module):

    def __init__(self,
                 in_feats : int,
                 out_feats : int) -> None:

        super().__init__()

        assert (in_feats >= 0), "In features is a quantity, it can't be negeative"
        assert (out_feats >= 0), "Out features is a quantity, it can't be negative"

        self.__in_feats : int = in_feats

        """ Parameters """
        stdv = 1. / sqrt(in_feats)
        self.__weights : Parameter = Parameter(autograd.functionnal.uniform_(-stdv, stdv, out_feats * in_feats).reshape(out_feats, in_feats))
        self.__biais : Parameter = Parameter(autograd.functionnal.uniform_(-stdv, stdv, out_feats).unsqueeze(0))

    
    def __forward__(self, x : autograd.jtensors.JTensors) -> autograd.jtensors.JTensors:

        assert (x is not None), "Invalid value passed as parameter"
        assert (x.shape() == (1, self.__in_feats) or x.shape() == (x.shape()[0], self.__in_feats)), "Invalid shape for input tensor. it may be :" + str((1, self.__in_feats)) + " or " + str((x.shape()[0], self.__in_feats)) + ", but got :" + str(x.shape())

        return autograd.functionnal.matmul(x, self.__weights.get().transpose()) + self.__biais.get()


    def __call__(self, x : autograd.jtensors.JTensors) -> autograd.jtensors.JTensors:
    
        return self.__forward__(x)


class Sequential(Module):

    def __init__(self, *modules : Module) -> None:
        super().__init__()

        self.modules : list[Module] = list(modules)


    def __call__(self, x : autograd.jtensors.JTensors) -> autograd.jtensors.JTensors:
        return functools.reduce(lambda acc, layer : layer(acc), self.modules, x)


class SoftMax(Module):

    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, x : autograd.jtensors.JTensors) -> Any:
        return autograd.functionnal.softmax(x)


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, x : autograd.jtensors.JTensors) -> Any:
        return autograd.functionnal.sigmoid(x)


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, x : autograd.jtensors.JTensors) -> Any:
        return autograd.functionnal.max(x, 0)


class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, x : autograd.jtensors.JTensors) -> Any:
        return autograd.functionnal.tanh(x)


class Dropout(Module):

    def __init__(self, p : float = 0.5) -> None:
        super().__init__()

        assert (p >= 0 and p <= 1), "Invalid value for p. p must be between 0 and 1"

        self.__p : float = p


    def __call__(self, x : autograd.jtensors.JTensors) -> Any:
        return autograd.functionnal.dropout(x, self.__p)


# TODO :
# look for lstm way of working till end of paper
# implement lstm
# save feature
# make adam optimizer
# add embedding layer


class RNN(Module):

    def __init__(self, input : int, 
                       hidden : int,
                       num_layers : int = 1,
                       non_linearity : str = 'tanh',
                       bias : bool = True,
                       batch_first : bool = False,
                       dropout : float = 0) -> None:
        """_summary_

        Args:
            input (int): input size
            hidden (int): hidden layer / context size (= output size)
            num_layers (int, optional): number of layers. Defaults to 1.
            non_linearity (str, optional): activation function. Defaults to 'tanh'.
            bias (bool, optional): add of a bias (linear to affine). Defaults to True.
            dropout (float, optional): neurone dropout avg. Defaults to 0.
        """

        super().__init__()

        assert (non_linearity == 'tanh' or non_linearity == 'relu'), "Invalid value for non_linearity. non_linearity must be 'tanh' or 'relu'"
        assert (dropout >= 0 and dropout <= 1), "Invalid value for dropout. dropout must be between 0 and 1"

        self._num_layers : int = num_layers
        self._hidden_size : int = hidden
        self._input_size : int = input
        self._non_linearity : str = non_linearity
        self._batch_first : bool = batch_first
        self._bias : bool = bias
        self._dropout : float = dropout
        self._dp_rule : bool = dropout > 0

        """ xavier initialization """
        for i in range(self._num_layers):

            stdv = 1. / sqrt(hidden)
            setattr(self, f'weight_ih_l{i}', Parameter(autograd.functionnal.uniform_(-stdv, stdv, hidden * input).reshape(input, hidden), f'weight_ih_l{i}'))
            if bias:
                setattr(self, f'bias_ih_l{i}', Parameter(autograd.functionnal.uniform_(-stdv, stdv, hidden).unsqueeze(0), f'bias_ih_l{i}'))

            setattr(self, f'weight_hh_l{i}', Parameter(autograd.functionnal.uniform_(-stdv, stdv, hidden * hidden).reshape(hidden, hidden), f'weight_hh_l{i}'))
            if bias:
                setattr(self, f'bias_hh_l{i}', Parameter(autograd.functionnal.uniform_(-stdv, stdv, hidden).unsqueeze(0), f'bias_hh_l{i}'))


    def __call__(self, x : autograd.jtensors.JTensors, hx : autograd.jtensors.JTensors = None) -> Any:

        assert (x is not None), "Invalid value passed as parameter"
        assert (len(x.shape()) == 3 or len(x.shape()) == 2), "Invalid shape for input tensor. tensor of shape (L,Hin)(L,Hin) for unbatched input, (L,N,Hin)(L,N,Hin) when batch_first=False or (N,L,Hin)(N,L,Hin) when batch_first=True"
        assert (x.shape()[-1] == self._input_size), "Invalid shape for input tensor. tensor of shape (L,Hin)(L,Hin) for unbatched input, (L,N,Hin)(L,N,Hin) when batch_first=False or (N,L,Hin)(N,L,Hin) when batch_first=True"
        assert (hx is None or (len(hx.shape()) == 2 and hx.shape()[0] == self._num_layers and hx.shape()[1] == self._hidden_size) or (len(hx.shape()) == 3 and hx.shape()[0] == self._num_layers and hx.shape()[1] == x.shape()[0] and hx.shape()[2] == self._hidden_size)), "Invalid shape for hidden"

        # multiply each element of the sequence one by one with the weights

        # init hidden states
        if hx is None:
            if len(x.shape()) == 3:
                shape = (self._num_layers, x.shape()[0], self._hidden_size) if self._batch_first else (self._num_layers, x.shape()[1], self._hidden_size)
                hx = autograd.functionnal.zeros(*shape)
            else:
                shape = (self._num_layers, self._hidden_size)
                hx = autograd.functionnal.zeros(*shape)

        hx.require_grads = True
        hx = autograd.functionnal.to_list(hx)

        output : list[autograd.jtensors.JTensors] = []

        if len(x.shape()) == 3 and self._batch_first:
            x = autograd.einops.rearrange(x, 'N L H -> L N H')
        if len(x.shape()) == 2:
            x = x.unsqueeze(1)

        seqlen = x.shape()[0]

        for seqi in range(seqlen):
            y = autograd.jtensors.JTensors(x[seqi], require_grads=True)
            for layer in range(self._num_layers):
                y = autograd.functionnal.matmul(y, self.__getattribute__(f'weight_ih_l{layer}').get())
                o = autograd.functionnal.matmul(hx[layer], self.__getattribute__(f'weight_hh_l{layer}').get())
                #print(y)
                #print(o)
                if self._bias:
                    y = y + self.__getattribute__(f'bias_ih_l{layer}').get()
                    o = o + self.__getattribute__(f'bias_hh_l{layer}').get()
                #print(y)
                # ht+1[layer] = yt[layer]
                hx[layer] = autograd.functionnal.tanh(y + o)
                #print(hx[layer])
                if self._dropout > 0:
                    hx[layer] = autograd.functionnal.dropout(hx[layer], self._dropout)
                y = hx[layer].clone()
            output.append(y)
        
        # return hidden state
        output : autograd.jtensors.JTensors = autograd.functionnal.from_list(output)
        if len(x.shape()) == 3 and self._batch_first:
            output = autograd.einops.rearrange(output, 'L N H -> N L H')
            #print("output shape", output.shape())
            #print("output strides", output.stride())
            #print(output)

        return output, autograd.functionnal.from_list(hx)


    def __str__(self) -> str:
        return f'JRNN(input={self._input_size}, hidden={self._hidden_size}, num_layers={self._num_layers}, non_linearity={self._non_linearity}, bias={self._bias}, dropout={self._dropout})'
    

    def __repr__(self) -> str:
        return self.__str__()