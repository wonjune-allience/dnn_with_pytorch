
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features=1, out_features=1)

print(list(model.parameters()))

class LR(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        out  = self.linear(x)
        return out

model = LR(1, 1)

print(list(model.parameters()))

x = torch.tensor([1.0])
yhat = model(x)

print(yhat)

# linear regression
def forward(x):
    return w * x

def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

class plot_diagram():
    
    # Constructor
    def __init__(self, X, Y, w, stop, go = False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values] 
        w.data = start
        
    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y,'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        temp = torch.FloatTensor(self.Loss_function)
        plt.plot(self.parameter_values.detach().numpy(), temp.detach().numpy())   
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
    
    # Destructor
    def __del__(self):
        plt.close('all')

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X
Y = f + 0.1 * torch.randn(X.size())


w = torch.tensor(-15.0, requires_grad=True)

LOSS2 = []
lr = 0.1

gredient_plot1 = plot_diagram(X, Y, w, stop=15)


def train_model(iter):

    for epoch in range(iter):

        yhat = forward(X)

        loss = criterion(yhat, Y)

        gredient_plot1(yhat, w, loss.item(), epoch)

        LOSS2.append(loss)
        loss.backward()

        w.data = w.data - lr * w.grad.data
        w.grad.data.zero_()


train_model(4)

tmp_LOSS2 = torch.FloatTensor(LOSS2)
plt.plot(tmp_LOSS2.detach().numpy())
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")



        