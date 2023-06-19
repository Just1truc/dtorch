
import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return np.tanh(x)
x = np.linspace(-10, 10, 100)
y = relu(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.title('Tanh Function')
plt.grid(True)
plt.savefig('tanh_plot.png')