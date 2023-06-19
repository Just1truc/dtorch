from dtorch import nn
from dtorch.jtensors import JTensors

class MyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.linear_layer1 = nn.Linear(1, 32)
        self.linear_layer2 = nn.Linear(32, 4)


    def forward(self, x : JTensors):
        return self.linear_layer2(self.linear_layer1(x))
    

# data
x = JTensors([2, 3, 4, 5])

# model
model = MyModel()

# call the model
# give a tensor of size (4,) as it's the output of the layer2
result = model(x)
