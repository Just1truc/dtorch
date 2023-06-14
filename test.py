from dtorchvision.models import AutoEncoder
from dtorchvision.datasets import MNISTDataset
from dtorch.optim import Adam
from dtorch.loss import MSELoss
from torchvision.datasets import MNIST
import torch as t
from dtorchvision.models import dt

class TorchAutoEncoder(t.nn.Module):

    def __init__(self,
                input : int,
                hidden_sizes : list[int],
                dp : float = 0.0) -> None:
        """AutoEncoders

        Args:
            input (int): entry
            hidden_sizes (list[int]): list of hidden sizes I -> C (input to encoded space)
            dp (float, optional): dropout. Defaults to 0.0.
        """

        super(TorchAutoEncoder, self).__init__()

        self.seq = t.nn.Sequential(
            t.nn.Linear(input, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, input),
            t.nn.ReLU()
        )

    
    def forward(self, x):
        return self.seq(x)
    
# torch_autoencoder = TorchAutoEncoder(784, [128, 128])
# data = MNISTDataset()
# train, test = data.data
# print(torch_autoencoder)

# optim = t.optim.Adam(torch_autoencoder.parameters(), lr=0.001)

# epochs = 10
# loss = t.nn.MSELoss()
# x, _ = train
# x = t.tensor(x.numpy(), dtype=t.float32)
# batched_x = x.split(32)
# for i in range(epochs):
#     #data = 0
#     #for x in batched_x:

#     y_pred = torch_autoencoder(x)
#     optim.zero_grad()
#     res = loss(y_pred, x)
#     print("Loss", res.item())
#     res.backward()
#     optim.step()
    
    #print("Loss", data)

autoencoder = AutoEncoder(784, [128, 32])
data = MNISTDataset()

train, test = data.data

optimizer = Adam(autoencoder.parameters(), lr = 0.001)

print(autoencoder)
epochs = 100
loss = MSELoss()
for i in range(epochs):

    x, _ = train

    y_pred = autoencoder(x)
    optimizer.zero_grad()

    res = loss(y_pred, x)
    print(f"Loss", res[0])
    res.backward()

    optimizer.step()

autoencoder.save('model.jt')
