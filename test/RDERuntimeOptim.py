from torch import tensor
from math import sqrt

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from RDEs import *
from ExamplePaths import *


def weak_error(approx_sol, exact_sol):
    return (approx_sol.mean(dim=0) - exact_sol.mean(dim=0)).abs()


def strong_error(approx_sol, exact_sol):
    return (approx_sol - exact_sol).abs().mean(dim=0)


batch_size = 1000
alpha = .1
beta = .05
x0 = .5
T = 1.


def mu(X, t):
    # X of shape [b x 1]
    # t of shape 1
    return (beta/(1 + t).sqrt()).view(1, 1).repeat(batch_size, 1) - 1/(2 * (1 + t)) * X


def f(X, t):
    val = alpha*beta/(1 + t).sqrt()
    return val.repeat(batch_size).view(batch_size, 1, 1)


f_prime = zeros(batch_size, 1, 1, 1)
path = ItoBrownianRoughPath(1, batch_size)


def correct_solution(t):
    return 1/sqrt(1 + t) * x0 + beta/sqrt(1 + t) * (t + alpha * path(t)[0])


N = 20000

del_t_max = T/N
sol = RDESolution(mu, f, 1, path, f_prime, starting_point=x0, delta_t_max=del_t_max)
approx_sol = sol(T, show_progress=False)
exact_sol = correct_solution(T)
print(f"N={N}\tmean_approx_sol={approx_sol.mean(dim=0)}\tmean_exact_sol={exact_sol.mean(dim=0)}")
print(f"weak_error={weak_error(approx_sol, exact_sol).item():.3e}\tstrong_error={strong_error(approx_sol, exact_sol).item():.3e}")
