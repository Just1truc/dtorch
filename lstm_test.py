""" imports """
import dtorch.nn as nn
import dtorch.optim as optim
import dtorch.jtensors as jt
import dtorch.loss as ls
import torch as t
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(15000)

sequence_length = 4

def load_dataset(filename : str):
    dataset = pd.read_csv(filename)
    data = np.array(list(dataset.date))
    return jt.JTensors(data - data[-1], require_grads=True), jt.JTensors((np.array(list(dataset.high)) + np.array(list(dataset.low))) / 2, require_grads=True)

x_train, y_train = load_dataset(sys.argv[1])

y_train_s = jt.JTensors(((y_train.numpy() / y_train.numpy().max())) - 0.7, require_grads=True)
x_train.update(np.lib.stride_tricks.as_strided((y_train_s() / y_train_s().max()), shape=(len(y_train_s) - 2, sequence_length), strides=(y_train_s().strides[0], y_train_s().strides[0])))
y_train.update(x_train[1:])
x_train.update(x_train[:-1])

# a tester
# w x y z => [[w, x, y], [x, y, z]]

x_train_tensor = x_train.reshape(553, sequence_length, 1)
y_train_tensor = y_train.reshape(553, sequence_length, 1)

x_torch = t.tensor(x_train.numpy(), dtype=t.float32, requires_grad=True).unsqueeze(-1)
y_torch = t.tensor(y_train.numpy(), dtype=t.float32, requires_grad=True).unsqueeze(-1)


class MyBalls(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.RNN(1, 1, 2, batch_first=True, bias=True)

    def forward(self, x):
            
        res, h = self.lstm(x)
        return res, h
    

class MyTorchBalls(t.nn.Module):
    def __init__(self):
        super(MyTorchBalls, self).__init__()
        self.rnn = t.nn.RNN(1, 1, 2, batch_first=True, bias=True)

    def forward(self, x):
        return self.rnn(x)

    
torch_model = MyTorchBalls()
model = MyBalls()
torch_model.rnn.weight_hh_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.weight_ih_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.bias_hh_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.bias_ih_l0 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.weight_hh_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.weight_ih_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.bias_hh_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
torch_model.rnn.bias_ih_l1 = t.nn.Parameter(t.tensor([[0.5]], requires_grad=True))
model.lstm.weight_hh_l1 = nn.Parameter(jt.JTensors([[0.5]]))
model.lstm.weight_ih_l1 = nn.Parameter(jt.JTensors([[0.5]]))
model.lstm.bias_hh_l1 = nn.Parameter(jt.JTensors([[0.5]]))
model.lstm.bias_ih_l1 = nn.Parameter(jt.JTensors([[0.5]]))
model.lstm.weight_hh_l0 = nn.Parameter(jt.JTensors([[0.5]]), 'weight_hh_l0')
model.lstm.weight_ih_l0 = nn.Parameter(jt.JTensors([[0.5]]), 'weight_ih_l0')
model.lstm.bias_hh_l0 = nn.Parameter(jt.JTensors([[0.5]]))
model.lstm.bias_ih_l0 = nn.Parameter(jt.JTensors([[0.5]]))

# i = input('load model ?')
# if i == 'y':
#     torch_model.load_state_dict(t.load('model.pt'))
#     torch_model.eval()
# 
#     m, _ = torch_model(x_torch)
# 
#     valid = 0
#     tot = 0
#     for i in range(len(m)):
#         o = m[i].item() - m[i - 1].item()
#         u = y_torch[i][-1].item() - y_torch[i - 1][-1].item()
#         if o < 0 and u < 0 or o > 0 and u > 0 or o == 0 and u == 0:
#             valid += 1
#         tot += 1
# 
#     print(valid / tot)
# 
#     plt.scatter(np.arange(0, len(x_train())), y_train_s()[:-3], label='Training data')
#     plt.plot(np.arange(0, len(x_train())), m.detach().numpy(), color='red', label='Predicted')
#     #plt.plot(x_train(), y_pred(), color='red', label='Predicted')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Non-linear Model Predicting Sine Function')
#     plt.legend()
#     plt.show()

#print(model.lstm._all_weights)
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer_torch = t.optim.SGD(torch_model.parameters(), lr=0.01)
#x = jt.JTensors([[1], [40], [2], [8]]) / 40
#x_torch = t.tensor([[[1], [40], [2], [8]]], dtype=t.float32) / 40
#x_train_tensor = jt.JTensors([[[2], [40]], [[1], [30]]]) / 40
#x_torch = t.tensor([[[2], [40]], [[1], [30]]], dtype=t.float32) / 40

#print(t.tanh(t.matmul(t.tensor([[[1], [3]]], dtype=t.float32), model.lstm.weight_ih_l0.t()) + model.lstm.bias_ih_l0 + t.matmul(t.tensor([[0]], dtype=t.float32), model.lstm.weight_hh_l0.t()) + model.lstm.bias_hh_l0))
loss = ls.MSELoss()
loss_torch = t.nn.MSELoss()
#y_train_tensor = jt.JTensors([[[40], [8]], [[30], [9]]]) / 40
#y_torch = t.tensor([[[40], [8]], [[30], [9]]], dtype=t.float32) / 40

a= 0
#print("y:", y_train_tensor)
epochs = 1
for i in range(epochs):
    optimizer.zero_grad()
    m, prout = model.forward(x_train_tensor)
    #print("m", m)

    optimizer_torch.zero_grad()
    m_torch, prout_torch = torch_model.forward(x_torch)
    #print("m_torch", m_torch)
    #print("prout:", prout)
    a = loss(m, y_train_tensor)
    print(f"loss at {i}:", a[0])
    a.backward()
    print("autograd:")
    print(model.lstm.weight_hh_l0.get().grad)
    print(model.lstm.weight_ih_l0.get().grad)
    optimizer.step()
    u_torch = loss_torch(m_torch, y_torch)
    u_torch.backward()
    print("torch:")
    print(torch_model.rnn.weight_hh_l0.grad)
    print(torch_model.rnn.weight_ih_l0.grad)
    optimizer_torch.step()
    print(f"loss_torch at {i}:", u_torch.item())

model.eval()
torch_model.eval()

m, prout = model.forward(x_train_tensor)
m_torch, prout_torch = torch_model.forward(x_torch)
#t.save(torch_model.state_dict(), "torch_model.pt")
#print("prout:", prout)
#print("prout_torch:", prout_torch)
print("m:", m)
print("m_torch:", m_torch)

# a b c => 
#m = ()

#valid = 0
#tot = 0
#for i in range(len(m_torch)):
#    #print("===")
#    #print(m_torch[i])
#    #print(y_train_tensor[i])
#    tot += 1
#    valid += (m_torch[i][-1].item() > x_torch[i][-1]) == (y_train_tensor[i][-1][0] > x_torch[i][-1])
#
#print("valid:", valid)
#print("tot:", tot)
#print("accuracy:", valid / tot)
#print(a)


#plt.scatter(np.arange(0, len(x_train())), y_train_s()[:-3], label='Training data')
#print(y_train_s()[-3:])
#print([x[2] for x in m.numpy()][:3])
#print(m.numpy()[-3:])
#plt.plot(np.arange(0, len(x_train())), [x[2] for x in m.numpy()], color='red', label='Predicted')
##plt.plot(x_train(), y_pred(), color='red', label='Predicted')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Non-linear Model Predicting Sine Function')
#plt.legend()
#plt.show()

#i = input("save model? (y/n): ")
#if i == "y":
#    t.save(torch_model.state_dict(), "./model.pt")
#    print("model saved")
#else:
#    print("model not saved")

# save the model
#i = input("save model? (y/n): ")
#if i == "y":
#    model.save("./model.jt")
#    print("model saved")
#else:
#    print("model not saved")


#print("autograd:")
#print(model.lstm.weight_hh_l0)
#print(model.lstm.weight_ih_l0)
#print("torch:")
#print(torch_model.lstm.weight_hh_l0)
#print(torch_model.lstm.weight_ih_l0)
