#!/usr/bin/env python
# Exercises using automatic differentiation

import torch
import numpy as np

# x = torch.Tensor([1, 2], requires_grad=True)

def g(x):
    return 2 * x + 1

def f(x):
    return x ** 2

x = torch.tensor([1.], requires_grad=True)
y = f(g(x))

x.grad # gradient hasn't appeared yet
y.backward()
# to evaluate at x = 0,
# y.backward(torch.tensor([0.]))
# can specify any values at input at which to take gradient

print(x.grad) ## dy / dx

# what about multidimensional inputs?
def g(x):
    return x[0] + 2 * x[1]

def f(x):
    return torch.sqrt(x)

x = torch.tensor([0., 1.], requires_grad=True)
y = f(g(x))

y.backward()
print(x.grad)

# This example is still simple enough that you can check it by hand
#
# (x0 + 2x1) ** (1/2)
# 1/2 * (x0 + 2x1) ^ (-1/2) . (1, 2)^T
# at [0, 1] this is [1/(2 * sqrt(2)) and 2/(2 * sqrt(2))]
# check that this is correct

# can even do crazy relu stuff (y = sigma(W * x))
def g(x):
    W = torch.randn((10, 10))
    return W.mm(x)

def f(x):
    return torch.clamp(x, min=0)

def h(x):
    return torch.sum(x)


x = torch.randn((10, 1), requires_grad=True)
y = h(f(g(x)))

y.backward()
print(x.grad)

# In theory, you could use these gradients to do neural network optimization by
# yourself, without ever referencing the optim class in pytorch
