from abc import ABC, abstractmethod
from torch import tensor, zeros, normal, ones, eye, Tensor, einsum, cat
from torch.autograd.functional import jacobian
from math import sqrt, ceil, exp
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from RoughPaths import *
from RDEs import *


batch_size = 10
n = 1
m = n
T = 1.

NN = nn.Sequential(nn.Linear(n + 1, 5), nn.LeakyReLU(), nn.Linear(5, 5), nn.LeakyReLU(), nn.Linear(5, m),
                   nn.LeakyReLU())


def mu(Y, t):
    inp = cat((Y, t * ones(Y.shape[0], 1)), dim=1)
    out = NN(inp)
    return out


def f(Y, t):
    return Y.unsqueeze(2)


B_t = ItoBrownianRoughPath(n, batch_size)
N = 10
optim = Adam(NN.parameters())
for i in range(N):
    optim.zero_grad()
    B_t.reset()
    sol = RDESolution(mu, f, m, B_t, starting_point=1., delta_t_max=0.01, for_backprop=True)

    loss = sol(T).mean(dim=0) ** 2 + sol(T).var(dim=0, unbiased=True)
    loss.backward()
    # print(f"Loss in epoch {i+1}: {loss}")
    optim.step()
