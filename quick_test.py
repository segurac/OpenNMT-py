import torch
from torch.autograd import Variable

x = torch.rand(100)
var = Variable(x)

# t = torch.gt(x, 0.2)
# t = t.float()

t = torch.rand(100)
var_t = Variable(t)

# y = var * var_t
y = var

z = y*y*3
out = z.mean()

out.backward()