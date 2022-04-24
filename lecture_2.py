import torch
#import numpy as nn

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