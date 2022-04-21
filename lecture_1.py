
import torch


import numpy as np
import matplotlib.pyplot as plt

#### 1-D tensor ####
c = torch.tensor([100, 1, 2, 3, 0])
print(c.dtype)
print(c)

v = c + 1
print(v)

x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)
plt.plot(x.numpy(), y.numpy())
#plt.show()

u = torch.tensor([1,2])
v = torch.tensor([0,1])
print(torch.dot(u,v))


#### 2-D tensor ####
X = torch.tensor([[1,1], [1,3], [3,4]])
print(X.shape)

A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)

print(A_times_B)

#### derivatives ####
