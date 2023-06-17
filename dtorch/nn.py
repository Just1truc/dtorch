""" imports """
from typing import Any, Optional
import dtorch
import functools
from math import sqrt
import pickle
from abc import ABC
import os
import requests

# gabriel destr

class JPretrainedModel(ABC):

    def __init__(self,
                 model_name : str,
                 type : str,
                 root : str = '.models') -> None:
        """Pretrained model interface

        Args:
            model_name (str): model name
            root (str, optional): where the model should be found. Defaults to '.models'.
        """

        self.__model_url : str = 'https://raw.githubusercontent.com/Just1truc/dtorch/main/pretrained_model/' + type + '/' + model_name + '.jt'
        self.__model_path : str = os.path.join(root, f'{model_name}.jt')
        self.__root : str = root

        if not(os.path.isdir(root)):
            os.makedirs(self.__root)
        
        if not(os.path.exists(self.__model_path)):
            with open(self.__model_path, 'wb') as bf:
                bf.write(requests.get(self.__model_url).content)


    @property
    def model_path(self):
        return self.__model_path


class Parameter:

    def __init__(self, tensor : dtorch.jtensors.JTensors, name : str = "") -> None:
        
        self.__tensor : dtorch.jtensors.JTensors = tensor
        self.__tensor.require_grads = True
        if (name != ""):
            self.__tensor.add_name(name)


    def get(self) -> dtorch.jtensors.JTensors:
        return self.__tensor
    

    def __str__(self) -> str:
        return "Parameter(" + self.__tensor.__str__() + ")"


    def __repr__(self) -> str:
        return self.__str__() 


class Module:

    def __init__(self) -> None:
        # transform submodules into dict for serialization
        #self._serialized_sub_modules : list = []
        self._sub_modules : list = []
        self._parameters : list[Parameter] = []

        #self.__last = None
        # count self
        #self._sub : int = 0


    """ private """

    def __setattr__(self, __name: str, __value: Any) -> None:
        
        if (isinstance(__value, Module) and __value != self):
            self._sub_modules.append(__value)

        if (isinstance(__value, list) and len(__value) > 0 and isinstance(__value[0], Module)):
            for module in __value:
                self._sub_modules.append(module)

        if (isinstance(__value, Parameter)):
            __value.get().add_name(__name)
            self._parameters.append(__value)
            #if (self.__last is not self):
            #    if (self.__last is not None):
            #        self._sub += 1
            #    self.__last = self
            #    self._serialized_sub_modules.append({})
#
            #self._serialized_sub_modules[self._sub][__name] = __value

        self.__dict__.__setitem__(__name, __value)


    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({', '.join([s.__str__() for s in self._sub_modules])})"


    def __repr__(self) -> str:
        return self.__str__()


    """ public """

    def forward(self):
        pass


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


    def parameters(self):
        res = self._parameters
        for module in self._sub_modules:
            res += module.parameters()
        return res
    

    def eval(self) -> None:
        """set the model in eval mode. Reccursivly set require_grads of parameters to false 
        """

        for parameter in self._parameters:
            parameter.get().require_grads = False
        for module in self._sub_modules:
            if (hasattr(self, "_dp_rule")):
                module._dp_rule = False
            module.eval()


    def train(self) -> None:
        """Set model in train mode. Require grad for all parameters
        """

        for parameter in self._parameters:
            parameter.get().require_grads = True
        for module in self._sub_modules:
            if (hasattr(self, "_dp_rule")):
                module._dp_rule = True
            module.train()


    def serialized(self) -> dict:

        modules = {
            "sub_modules" : {},
            "params" : {}
        }
        i = 0
        for module in self._sub_modules:
            modules["sub_modules"][i] = module.serialized()
            i += 1
        for param in self._parameters:
            modules["params"][param.get().get_name()] = param
        
        return modules
    

    def load_serialized(self, saved_data : dict):

        for key, value in saved_data["sub_modules"].items():
            self._sub_modules[key].load_serialized(value)
        self._parameters = []
        for key, value in saved_data["params"].items():
            self.__setattr__(key, value)

    
    def save(self, path : str) -> None:
        """Save the model in the given path
        """
        # Reccursive serialization

        pickle.dump(self.serialized(), open(path, "wb"))
        
        # pickle.dump(self._serialized_sub_modules, open(path, "wb"))


    def load(self, path : str) -> None:
        """Load the model from the given path
        """
        # reccursivly loading

        try:
            self.load_serialized(pickle.load(open(path, 'rb')))
        except:
            raise Exception("Save file is not compatible with current model")

        #try:
        #    self._serialized_sub_modules = pickle.load(open(path, "rb"))
        #    # sequential saving does not work
        #    # clean current model
        #    Module._parameters = []
        #    Module._serialized_sub_modules = []
        #    Module._sub = 0
        #    Module.__last = None
        #    # load new model
        #    for id, sub_module in enumerate(self._serialized_sub_modules):
        #        for name, param in sub_module.items():
        #            Module._sub_modules[id].__setattr__(name, param)
        #except:
        #    raise Exception("Save file is not compatible with current model")


class Linear(Module):

    def __init__(self,
                 in_feats : int,
                 out_feats : int) -> None:

        super().__init__()

        assert (in_feats >= 0), "In features is a quantity, it can't be negeative"
        assert (out_feats >= 0), "Out features is a quantity, it can't be negative"

        self.__in_feats : int = in_feats
        self.__out_feats : int = out_feats

        """ Parameters """
        stdv = 1. / sqrt(in_feats)
        self.__weights : Parameter = Parameter(dtorch.functionnal.uniform_(-stdv, stdv, out_feats * in_feats).reshape(out_feats, in_feats))
        self.__biais : Parameter = Parameter(dtorch.functionnal.uniform_(-stdv, stdv, out_feats).unsqueeze(0))

    
    def forward(self, x : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:

        assert (x is not None), "Invalid value passed as parameter"
        assert (x.shape == (1, self.__in_feats) or x.shape == (x.shape[0], self.__in_feats)), "Invalid shape for input tensor. it may be :" + str((1, self.__in_feats)) + " or " + str((x.shape[0], self.__in_feats)) + ", but got :" + str(x.shape)

        return dtorch.functionnal.matmul(x, self.__weights.get().transpose()) + self.__biais.get()


    def __str__(self) -> str:
        return f"Linear ({self.__in_feats}, {self.__out_feats})"
    

    def __repr__(self) -> str:
        return self.__str__()


class Sequential(Module):

    def __init__(self, *modules : Module) -> None:
        super().__init__()

        self.modules : list[Module] = list(modules)


    def forward(self, x : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:
        return functools.reduce(lambda acc, layer : layer(acc), self.modules, x)
    

    def __str__(self) -> str:
        return f"Sequential ({', '.join([module.__str__() for module in self.modules])})"


    def __repr__(self) -> str:
        return self.__str__()


class SoftMax(Module):

    def __init__(self) -> None:
        super().__init__()

    
    def forward(self, x : dtorch.jtensors.JTensors) -> Any:
        return dtorch.functionnal.softmax(x)
    

    def __str__(self) -> str:
        return "SoftMax"
    

    def __repr__(self) -> str:
        return self.__str__()


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, x : dtorch.jtensors.JTensors) -> Any:
        return dtorch.functionnal.sigmoid(x)


    def __str__(self):
        return "Sigmoid"
    

    def __repr__(self) -> str:
        return self.__str__()


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, x : dtorch.jtensors.JTensors) -> Any:
        return dtorch.functionnal.max(x, 0)


    def __str__(self) -> str:
        return "ReLU"
    

    def __repr__(self) -> str:
        return self.__str__()


class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, x : dtorch.jtensors.JTensors) -> Any:
        return dtorch.functionnal.tanh(x)
    

    def __str__(self) -> str:
        return "Tanh"
    

    def __repr__(self) -> str:
        return self.__str__()


class Dropout(Module):

    def __init__(self, p : float = 0.5) -> None:
        super().__init__()

        assert (p >= 0 and p <= 1), "Invalid value for p. p must be between 0 and 1"

        self.__p : float = p


    def forward(self, x : dtorch.jtensors.JTensors) -> Any:
        return dtorch.functionnal.dropout(x, self.__p)
    

    def __str__(self) -> str:
        return f"Dropout({self.__p})"
    

    def __repr__(self) -> str:
        return self.__str__()


class Conv1d(Module):

    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 kernel_size : int,
                 stride : Optional[int] = 1,
                 bias : bool = True) -> None:
        """ Convolution layer

        Args:
            in_channels (int) : input size
            out_channels (int) : output size
            kernel_size (int) : size of the window moving
            stride (Optional[int]) : define the step of the window
        """

        super().__init__()

        assert (in_channels >= 0), "In channels is a quantity, it can't be negeative"
        assert (out_channels >= 0), "Out channels is a quantity, it can't be negative"
        assert (kernel_size >= 0), "Kernel size is a quantity, it can't be negative"
        assert (stride >= 0), "Stride is a quantity, it can't be negative"

        self.__stride : int = stride
        self.__in_c : int = in_channels
        self.__out_c : int = out_channels
        self.__ks : int = kernel_size
        self.__bias : bool = bias

        """ Parameters """

        self.bb : bool = bias
        self.biais : Parameter = None
        self.weights : Parameter = Parameter(dtorch.functionnal.xavier(kernel_size * in_channels, kernel_size * out_channels * in_channels).reshape(out_channels, in_channels, kernel_size))
        if bias:
            self.biais : Parameter = Parameter(dtorch.functionnal.xavier(kernel_size, out_channels))


    def forward(self, x : dtorch.jtensors.JTensors) -> dtorch.jtensors.JTensors:

        return dtorch.functionnal.conv1d(x, self.weights.get(), self.biais.get() if self.bb else None, self.__stride)


    def __str__(self) -> str:
        return f"Conv1d (in_channel = {self.__in_c}, out_channel = {self.__out_c}, kernel_size = {self.__ks}, bias = {self.__bias})"
    

    def __repr__(self) -> str:
        return self.__str__()


class RNN(Module):

    def __init__(self, input : int, 
                       hidden : int,
                       num_layers : int = 1,
                       non_linearity : str = 'tanh',
                       bias : bool = True,
                       batch_first : bool = False,
                       dropout : float = 0) -> None:
        """Reccurent neural network module

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
            setattr(self, f'weight_ih_l{i}', Parameter(dtorch.functionnal.uniform_(-stdv, stdv, hidden * input).reshape(input, hidden), f'weight_ih_l{i}'))
            if bias:
                setattr(self, f'bias_ih_l{i}', Parameter(dtorch.functionnal.uniform_(-stdv, stdv, hidden).unsqueeze(0), f'bias_ih_l{i}'))

            setattr(self, f'weight_hh_l{i}', Parameter(dtorch.functionnal.uniform_(-stdv, stdv, hidden * hidden).reshape(hidden, hidden), f'weight_hh_l{i}'))
            if bias:
                setattr(self, f'bias_hh_l{i}', Parameter(dtorch.functionnal.uniform_(-stdv, stdv, hidden).unsqueeze(0), f'bias_hh_l{i}'))


    def forward(self, x : dtorch.jtensors.JTensors, hx : dtorch.jtensors.JTensors = None) -> Any:

        assert (x is not None), "Invalid value passed as parameter"
        assert (len(x.shape) == 3 or len(x.shape) == 2), "Invalid shape for input tensor. tensor of shape (L,Hin)(L,Hin) for unbatched input, (L,N,Hin)(L,N,Hin) when batch_first=False or (N,L,Hin)(N,L,Hin) when batch_first=True"
        assert (x.shape[-1] == self._input_size), "Invalid shape for input tensor. tensor of shape (L,Hin)(L,Hin) for unbatched input, (L,N,Hin)(L,N,Hin) when batch_first=False or (N,L,Hin)(N,L,Hin) when batch_first=True"
        assert (hx is None or (len(hx.shape) == 2 and hx.shape[0] == self._num_layers and hx.shape[1] == self._hidden_size) or (len(hx.shape) == 3 and hx.shape[0] == self._num_layers and hx.shape[1] == x.shape[0] and hx.shape[2] == self._hidden_size)), "Invalid shape for hidden"

        # multiply each element of the sequence one by one with the weights

        # init hidden states
        if hx is None:
            if len(x.shape) == 3:
                shape = (self._num_layers, x.shape[0], self._hidden_size) if self._batch_first else (self._num_layers, x.shape[1], self._hidden_size)
                hx = dtorch.functionnal.zeros(*shape)
            else:
                shape = (self._num_layers, self._hidden_size)
                hx = dtorch.functionnal.zeros(*shape)

        hx.require_grads = True
        hx = dtorch.functionnal.to_list(hx)

        output : list[dtorch.jtensors.JTensors] = []

        if len(x.shape) == 3 and self._batch_first:
            x = dtorch.einops.rearrange(x, 'N L H -> L N H')
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        seqlen = x.shape[0]

        for seqi in range(seqlen):
            y = dtorch.jtensors.JTensors(x[seqi], require_grads=True)
            for layer in range(self._num_layers):
                y = dtorch.functionnal.matmul(y, self.__getattribute__(f'weight_ih_l{layer}').get())
                o = dtorch.functionnal.matmul(hx[layer], self.__getattribute__(f'weight_hh_l{layer}').get())
                #print(y)
                #print(o)
                if self._bias:
                    y = y + self.__getattribute__(f'bias_ih_l{layer}').get()
                    o = o + self.__getattribute__(f'bias_hh_l{layer}').get()
                #print(y)
                # ht+1[layer] = yt[layer]
                hx[layer] = dtorch.functionnal.tanh(y + o)
                #print(hx[layer])
                if self._dropout > 0:
                    hx[layer] = dtorch.functionnal.dropout(hx[layer], self._dropout)
                y = hx[layer].clone()
            output.append(y)
        
        # return hidden state
        output : dtorch.jtensors.JTensors = dtorch.functionnal.from_list(output)
        if len(x.shape) == 3 and self._batch_first:
            output = dtorch.einops.rearrange(output, 'L N H -> N L H')
            #print("output shape", output.shape)
            #print("output strides", output.stride)
            #print(output)

        return output, dtorch.functionnal.from_list(hx)


    def __str__(self) -> str:
        return f'JRNN(input={self._input_size}, hidden={self._hidden_size}, num_layers={self._num_layers}, non_linearity={self._non_linearity}, bias={self._bias}, dropout={self._dropout})'
    

    def __repr__(self) -> str:
        return self.__str__()