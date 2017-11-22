import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

x = np.zeros(300)
x.fill(3)
y = torch.from_numpy(x)
y = y.long()
x = torch.LongTensor([[[1,2,3],[4,5,6]]])

x = x.view(3,2,1)

print(x)
print(x.size())

a = x.view(x.size(0), -1)
print(a)

aux = x.size()
print(aux[0])


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # make a sequence of length 5

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(
    torch.randn((1, 1, 3))))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
