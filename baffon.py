import dtorch as dt
from dtorchvision.models import AutoEncoder
from dtorchvision.datasets import MNISTDataset
import dtorchvision.models
from matplotlib import pyplot as plt
import random

#autoencoder = AutoEncoder(784, [128, 32])
autoencoder = dtorchvision.models.MNISTAutoEncoder_128_32()

i = input("Train? (y/n) ")
if (i == 'n'):

    dataset = MNISTDataset(download=True)

    (x, _), _ = dataset.data

    a = autoencoder(x)
    loss = dt.loss.MSELoss()
    print(loss(a, x))
    img = random.randint(0, len(a) - 1)
    plt.imshow(a[img].reshape(28, 28))
    plt.show()
    plt.imshow(x[img].reshape(28, 28))
    plt.show()
    exit()

autoencoder.load('model.jt')
autoencoder.train()

dataset = MNISTDataset()

(x, _), _ = dataset.data

optimizer = dt.optim.Adam(autoencoder.parameters(), lr = 0.001)

print(autoencoder)
epochs = 1000
loss = dt.loss.MSELoss()
for i in range(epochs):

    y_pred = autoencoder(x)
    optimizer.zero_grad()

    res = loss(y_pred, x)
    print(f"Loss", res[0])
    res.backward()

    optimizer.step()

autoencoder.save('model.jt')
