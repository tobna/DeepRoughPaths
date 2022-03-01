from RoughPaths import *
from math import ceil
from torch.autograd.functional import jacobian
from tqdm import tqdm
from torch import is_tensor


def _e(i, n, batch_size=None):
    """
    cannonical basis vector
    :param i: index for 1 entry
    :param n: dimensions
    :param batch_size: batch_size or None
    :return: e^(i) [(b x) n]
    """
    if batch_size is not None:
        e = zeros(batch_size, n)
        e[:, i] = ones(batch_size)
        return e
    e = zeros(n)
    e[i] = 1
    return e


def _approx_jacobian_x(f, x, t, h):
    """
    Approximation of the jacobian, via difference quotient (differentiable)

    :param f: (x, t) -> [b x m x n]
    :param x: x [b x a]
    :param t: time [1 D]
    :param h: difference [1 D]
    :return: D_x f (x, t) [b x m x n x a]
    """
    batch_size, n = x.shape
    return cat([(f(x + h * _e(j, n, batch_size), t) - f(x - h * _e(j, n, batch_size), t)).unsqueeze(3) / (2 * h) for j in range(n)], dim=3)


class ConstantControlledPath(ControlledPath):
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

    def __call__(self, t):
        return self.f, self.f_prime


class RDESolution:
    """
    Solution to the RDE dY = mu(Y, t) dt + f(Y) dx
    with a rough path x
    """

    def __init__(self, drift, f, m, path: RoughPath, f_prime="difference_quotient", starting_point=None, delta_t_max=0.001):
        """
        Solution to the RDE dY = mu(Y, t) dt + f(Y) dx
        with a rough path x

        :param drift: mu: (Y [b x m], t [1]) |-> mu(Y, t) [b x m]
        :param f: Y [b x m] |-> f(Y) [b x m x n]
        :param path: Rough path x
        :param f_prime: Y [b x m], t[1] |-> D_Y f [b x m x n x m] OR 'difference_quotient', 'exact_no_grads', 'exact_with_grads' for automatic computation OR constant tensor [b x m x n x m]
        :param starting_point: Y_0
        :param delta_t_max: max delta t for approximation
        """
        self.drift = drift
        self.f = f
        assert callable(f_prime) or is_tensor(f_prime) or f_prime in ["difference_quotient", "exact_no_grads", "exact_with_grads"],\
            f"prime has to be a function, a tensor, or computation strategy, but got {f_prime}"
        if is_tensor(f_prime):
            assert f_prime.shape[0] == path.batch_size and f_prime.shape[1] == f_prime.shape[3] == m\
                and f_prime.shape[2] == path.n, f"Constant f_prime needs to be tensor of shape [b x m x n x m]" \
                                                f"= [{path.batch_size}, {m}, {path.n}, {m}], but got {f_prime.shape}."
        self.f_prime = f_prime
        self.path = path
        if starting_point is None:
            starting_point = zeros(path.batch_size, m)
        if isinstance(starting_point, int) or isinstance(starting_point, float):
            starting_point = starting_point * ones(path.batch_size, m)
        assert len(starting_point.shape) == 2 and starting_point.shape[1] == m \
               and starting_point.shape[0] == path.batch_size, \
            f"Need starting point of shape (batch_size={path.batch_size}, m={m}), " \
            f"but got {starting_point.shape} instead"
        self.starting_point = starting_point
        self.values = {0.: starting_point}
        self.delta_t_max = delta_t_max
        self.m = m

    def y_prime(self, t):
        # f(Y)
        Y_t = self(t)
        return self.f(Y_t, t)

    def __call__(self, t, show_progress=False):
        assert t >= 0, f"Only positive time values allowed, but got {t}"
        if t in self.values.keys():
            return self.values[t]

        t_max = max(self.values.keys())
        if t < t_max:
            # linear interpolation
            t_before = max([s for s in self.values.keys() if s < t])
            t_after = min([s for s in self.values.keys() if s > t])
            return (t - t_before) / (t_after - t_before) * self.values[t_after] + (t_after - t) / (t_after - t_before) * \
                   self.values[t_before]

        steps = ceil((t - t_max) / self.delta_t_max)
        delta_t = (t - t_max) / steps
        Y_last_t_i = self.values[t_max]
        last_t_i = t_max
        iteration = [delta_t * (i + 1) + t_max for i in range(steps)]
        if show_progress:
            iteration = tqdm(iteration)
        for t_i in iteration:
            mu = self.drift(Y_last_t_i, last_t_i)  # b x m
            sigma = Y_prime_last_t = self.f(Y_last_t_i, last_t_i)
            if self.f_prime == "difference_quotient":
                grad_f_Y = _approx_jacobian_x(self.f, Y_last_t_i, tensor(last_t_i), self.delta_t_max / 2)
            elif self.f_prime == "exact_no_grads":
                grad_f_Y = einsum('abcad -> abcd',
                                  jacobian(self.f, (Y_last_t_i, tensor(last_t_i)), vectorize=True)[0])  # b x m x n x m
            elif self.f_prime == "exact_with_grads":
                grad_f_Y = einsum('abcad -> abcd', jacobian(self.f, (Y_last_t_i, tensor(last_t_i)), vectorize=True, create_graph=True)[0])  # b x m x n x m
            elif is_tensor(self.f_prime):
                grad_f_Y = self.f_prime
            else:
                grad_f_Y = self.f_prime(Y_last_t_i, tensor(last_t_i))

            # batched matmul
            f_Y_prime = einsum('bijk,bkl -> bijl', grad_f_Y, Y_prime_last_t)  # b x m x n x n
            delta_controlled_path = ConstantControlledPath(sigma, f_Y_prime)
            Y_t_i = Y_last_t_i + mu * delta_t + self.path.rough_integral(delta_controlled_path, t_i, last_t_i, n=1)
            self.values[t_i] = Y_t_i
            last_t_i = t_i
            Y_last_t_i = Y_t_i
        return Y_last_t_i
