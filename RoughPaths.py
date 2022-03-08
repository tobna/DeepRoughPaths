from abc import ABC, abstractmethod
from torch import tensor, zeros, normal, ones, eye, Tensor, einsum, cat, zeros_like, ones_like
from torch.autograd.functional import jacobian
from torch.linalg import cholesky
from math import sqrt, ceil, factorial, floor
from tqdm import tqdm
from itertools import product
import string


_poss_einsum_indices = string.ascii_lowercase[0] + string.ascii_lowercase[2:]


def _tensor_log(X):
    assert (X[0] == ones_like(X[0])).all(), f"First component has to be 1, but was {X[0]}"
    max_level = len(X) - 1
    assert max_level <= len(_poss_einsum_indices), f"Implementation can only handle tensors with maximal level of " \
                                                   f"{len(_poss_einsum_indices)}, but got {max_level}"
    is_batched = len(X[0].shape) > 0
    if max_level <= 0:
        return [zeros_like(X[0])]
    log_X = [zeros_like(x) if i == 0 else x.clone() for i, x in enumerate(X)]
    X_tensor_n_last = [zeros_like(x) if i == 0 else x.clone() for i, x in enumerate(X)]
    for n in range(2, max_level + 1):
        factor = (-1)**(n + 1)/n
        X_tensor_n = len(X) * [0]
        for i, j in product(range(1, max_level+1), repeat=2):
            if i + j > max_level:
                continue
            if is_batched:
                einsum_str = _poss_einsum_indices[:j]
                X_tensor_n[i + j] += einsum(f'b...,b{einsum_str} -> b...{einsum_str}', X_tensor_n_last[i], X[j])
            else:
                einsum_str = _poss_einsum_indices[:j]
                X_tensor_n[i + j] += einsum(f'b...,{einsum_str} -> b...{einsum_str}', X_tensor_n_last[i], X[j])
        for i, Xi in enumerate(X_tensor_n):
            log_X[i] += factor * Xi
    return log_X


def _tensor_exp(X):
    max_level = len(X) - 1
    is_batched = len(X[0].shape) > 0
    if max_level <= 0:
        return [X[0].exp()]
    exp_X = [ones_like(x) if i == 0 else x.clone() for i, x in enumerate(X)]
    X_tensor_n_last = [x.exp() if i == 0 else x.clone() for i, x in enumerate(X)]
    for n in range(2, max_level + 1):
        factor = X[0].exp()[0] / factorial(n)
        X_tensor_n = len(X) * [0]
        for i, j in product(range(1, max_level + 1), repeat=2):
            if i + j > max_level:
                continue
            if is_batched:
                einsum_str = _poss_einsum_indices[:j]
                X_tensor_n[i + j] += einsum(f'b...,b{einsum_str} -> b...{einsum_str}', X_tensor_n_last[i], X[j])
            else:
                einsum_str = _poss_einsum_indices[:j]
                X_tensor_n[i + j] += einsum(f'b...,{einsum_str} -> b...{einsum_str}', X_tensor_n_last[i], X[j])
        for i, Xi in enumerate(X_tensor_n):
            exp_X[i] += factor * Xi
    return exp_X


class RoughPath(ABC):
    def __init__(self, n, batch_size, sig_cache_size=10):
        assert sig_cache_size > 0, f"Need cache of at least 1, but got {sig_cache_size}"
        self.n = n
        self.batch_size = batch_size
        self.eval_points = {0.}
        self.signature_vals = []
        self.sig_cache_size = sig_cache_size + 1

    @property
    @abstractmethod
    def p(self):
        pass

    @abstractmethod
    def __call__(self, t):
        """

        :param t: Time
        :return: x1_t (b x n), x2_t (b x n x n); x2_t(b, i ,j) = int x1_tau(b, i) dx1_tau(b, j)
        """
        assert t >= 0, f"Time t needs to be positive, but got {t}"
        self.eval_points.add(t)

    def rough_integral(self, controlled_path, t, s=0, n=100, progress=False):
        val = 0
        x1_last, x2_last = self(s)
        iterator = range(n)
        if progress:
            iterator = tqdm(iterator)
        for i in iterator:
            tau = (i + 1) / n * (t - s) + s
            Y_tau, Y_prime_tau = controlled_path(tau)
            x1_tau, x2_tau = self(tau)
            x1_delta = x1_tau - x1_last
            x2_delta = x2_tau - x2_last - einsum('ab,ac -> abc', x1_last, x1_delta)
            if isinstance(Y_prime_tau, Tensor):
                assert len(Y_prime_tau.shape) == 4 and Y_prime_tau.shape == \
                       (x1_tau.shape[0], Y_tau.shape[1], x1_tau.shape[1], x1_tau.shape[1]), \
                    f"Need tensor of dimensions {(x1_tau.shape[0], Y_tau.shape[1], x1_tau.shape[1], x1_tau.shape[1])}" \
                    f", got {Y_prime_tau.shape} instead"
                # einsum for batched tensor product
                Y_prime_tau_func = lambda x: einsum('bmij, bji -> bm', Y_prime_tau, x)
            else:
                Y_prime_tau_func = Y_prime_tau
            # einsum for batched matrix vector product
            val += einsum('bmi,bi -> bm', Y_tau, x1_delta) + Y_prime_tau_func(x2_delta)
            x1_last, x2_last = x1_tau, x2_tau
        return val

    def reset(self):
        self.eval_points = {0.}
        self.signature_vals = []

    def sig(self, t, N, delta_t_max=0.001, show_progress=False):
        N = N + 1
        assert 3 > self.p >= 2
        assert t >= 0.
        x1_t, x2_t = self(t)
        sig_list = [ones(self.batch_size), x1_t, x2_t]
        if N <= 3:
            return sig_list[:N]

        admissible_t_i = [t_i for t_i in sorted(self.eval_points) if t_i <= t]

        # ensure maximum delta_t
        if delta_t_max is not None:
            add_eval_points = []
            for s, t in zip(admissible_t_i[:-1], admissible_t_i[1:]):
                assert t > s
                if t - s > delta_t_max:
                    steps = ceil((t - s)/delta_t_max)
                    interm_steps = steps - 1
                    step_size = (t - s)/steps
                    add_eval_points += [step_size*(i+1) + s for i in range(interm_steps)]

            if len(add_eval_points) > 0:
                # do not remove intermediate calculations, even when delta_t > delta_t_max
                admissible_t_i = sorted(admissible_t_i + add_eval_points)

        # setup starting points
        if len(self.signature_vals) < N-3:
            # reset to starting point = 0
            for n in range(3, N):
                sig_vals_idx = n - 3
                sig_shape = [self.batch_size] + n * [self.n]
                if len(self.signature_vals) <= sig_vals_idx:
                    self.signature_vals.append({0.: zeros(*sig_shape)})
                else:
                    self.signature_vals[sig_vals_idx] = {0.: zeros(*sig_shape)}
            t_start = 0.
        else:
            possible_start_indices = set(self.signature_vals[0].keys())
            for n in range(4, N):
                sig_vals_idx = n - 3
                possible_start_indices = possible_start_indices.intersection(self.signature_vals[sig_vals_idx].keys())
            t_start = max([tau for tau in possible_start_indices if tau <= t])

        last_t_i = t_start
        x1_last, x2_last = self(last_t_i)
        iterator = [tau for tau in admissible_t_i if tau > t_start]
        if show_progress:
            iterator = tqdm(iterator)
        sig_vals_last_t_i = [sig_list[0]] + [x1_last, x2_last] + [sig_val[t_start] for sig_val in self.signature_vals]
        for t_i in iterator:
            x1_t_i, x2_t_i = self(t_i)
            x1_delta = x1_t_i - x1_last
            x2_delta = x2_t_i - x2_last - einsum('ab,ac -> abc', x1_last, x1_delta)
            sig_vals_t_i = [sig_list[0], x1_t_i, x2_t_i]
            for n in range(3, N):
                sig_val_t_i_lv_n = sig_vals_last_t_i[n] + einsum('b...,bj -> b...j', sig_vals_last_t_i[n-1], x1_delta) \
                                   + einsum('b...,bij -> b...ij', sig_vals_last_t_i[n-2], x2_delta)
                sig_vals_t_i.append(sig_val_t_i_lv_n)
            sig_vals_last_t_i = sig_vals_t_i

            last_t_i = t_i
            x1_last, x2_last = x1_t_i, x2_t_i

        for n, sig_val_t in enumerate(sig_vals_last_t_i):
            if n < 3:
                continue
            sig_vals_idx = n - 3
            self.signature_vals[sig_vals_idx][last_t_i] = sig_vals_last_t_i[n]

        if len(self.signature_vals[sig_vals_idx]) > self.sig_cache_size:
            # cache getting too large; remove the smallest cached time > 0
            self.signature_vals[sig_vals_idx].pop(min([tau for tau in self.signature_vals[sig_vals_idx].keys()
                                                       if tau > 0.]))

        for n in range(3, N):
            sig_vals_idx = n - 3
            sig_list.append(self.signature_vals[sig_vals_idx][t])
        return sig_list

    def log_sig(self, t, N, delta_t_max=0.001, show_progress=False):
        sig = self.sig(t, N, delta_t_max=delta_t_max, show_progress=show_progress)
        return _tensor_log(sig)


class ControlledPath(ABC):
    @abstractmethod
    def __call__(self, t):
        """
        Evaluation of the controlled rough path.

        :param t: Time
        :return: Y_t = a Matrix, Y'_t = a linear function R^{batch x n x n} -> R^{batch x m} (using tensordot)
        """
        pass


class FunctionControlledPath(ControlledPath):
    """
    Controlled Path for a function f: R^n -> L(R^n, R^m)
    """

    def __init__(self, f, x, for_backprop=False):
        """
        :param f: Function to calculate f(x); R^{batch x n} -> R^{batch x m x n}
        :param x: Underlying rough path to calculate x(t)
        """
        self.f = f
        self.x = x
        self.create_graph = for_backprop

    def __call__(self, t):
        super(FunctionControlledPath, self).__call__(t)
        x_t = self.x(t)[0]
        # take out additional batch dimensions
        grad_f_x_t = einsum('abcad -> abcd', jacobian(self.f, x_t, vectorize=True, create_graph=self.create_graph))
        return self.f(x_t), grad_f_x_t


class FTime(ControlledPath):
    def __init__(self, n, batch_size=1):
        self.n = n
        self.batch_size = batch_size

    def __call__(self, t):
        super(FTime, self).__call__(t)
        return t * eye(self.n).repeat(self.batch_size, 1, 1), zeros(self.batch_size, self.n, self.n, self.n)


class ExtendedRoughPath(RoughPath):
    def __init__(self, path: RoughPath, delta_t=0.01, approx_x2_cache_size=None):
        super(ExtendedRoughPath, self).__init__(path.n + 1, path.batch_size)
        self.path = path
        self.delta_t = delta_t
        self.approx_x2_cache_size = None if approx_x2_cache_size is None else approx_x2_cache_size + 1
        self.x2_parent_wrt_time = {0.: zeros(self.batch_size, self.n - 1)}

    @property
    def p(self):
        return self.path.p

    def approx_x2(self, t):
        if t == 0:
            return zeros(self.batch_size, self.n, self.n)

        # i, j < n-1 => take values from parent path
        x1_t_parent, x2_t_parent = self.path(t)  # b x n-1 x n-1

        # i = n-1, j < n-1 => int x^i_tau dtau
        max_t_prior = max(self.x2_parent_wrt_time.keys())
        min_t_prior = min([tau for tau in self.x2_parent_wrt_time.keys() if tau > 0.]) if len(self.x2_parent_wrt_time.keys()) > 1 else 0.
        if t in self.x2_parent_wrt_time.keys():
            x2_t_parent_wrt_time = self.x2_parent_wrt_time[t]
        elif min_t_prior <= t <= max_t_prior:
            # linear interpolation
            t_before = max([tau for tau in self.x2_parent_wrt_time.keys() if tau < t])
            t_after = min([tau for tau in self.x2_parent_wrt_time.keys() if tau > t])
            x2_t_parent_wrt_time = (t - t_before) / (t_after - t_before) * self.x2_parent_wrt_time[t_after] + (
                        t_after - t) / (t_after - t_before) * self.x2_parent_wrt_time[t_before]
        else:
            t_start = 0. if t < min_t_prior else max_t_prior
            steps = ceil((t - t_start) / self.delta_t)
            delta_t = (t - t_start) / steps
            x2_t_parent_wrt_time = self.x2_parent_wrt_time[t_start] + delta_t * sum([self.path(delta_t * i + t_start)[0] for i in range(steps)]) + (self.path(t)[0] - self.path(t_start)[0]) / 2
            self.x2_parent_wrt_time[t] = x2_t_parent_wrt_time
            if self.approx_x2_cache_size is not None and len(self.x2_parent_wrt_time) > self.approx_x2_cache_size:
                self.x2_parent_wrt_time.pop(min([tau for tau in self.x2_parent_wrt_time.keys() if tau > 0.]))

        # i < n-1, j = n-1 => int tau dx^i_tau
        # x2_t_time_wrt_parent = self.path.rough_integral(FTime(self.n - 1, self.batch_size), t, s=0,
        #                                                n=ceil(t / self.delta_t))  # b (x 1) x n-1
        # we can also use the integration by parts rule / shuffle product, to see int tau dx^i_tau = (x^i_t - x^i_s) (t - s) - int x^i_tau dtau
        x2_t_time_wrt_parent = t * x1_t_parent - x2_t_parent_wrt_time

        # i = n-1, j = n-1 => int tau dtau = t^2/2
        x2_t_time_wrt_time = t ** 2 / 2 * ones(self.batch_size, 1, 1)  # b x 1 x 1

        # correct order from:
        # x2_t(b, i ,j) = int x1_tau(b, i) dx1_tau(b, j)

        inter_1 = cat((x2_t_parent, x2_t_parent_wrt_time.unsqueeze(2)), dim=2)  # b x n-1 x n
        inter_2 = cat((x2_t_time_wrt_parent.unsqueeze(1), x2_t_time_wrt_time), dim=2)  # b x 1 x n

        return cat((inter_1, inter_2), dim=1)  # b x n x n

    def __call__(self, t):
        super(ExtendedRoughPath, self).__call__(t)
        if self.p < 2:
            return cat((self.path(t), t * ones(self.batch_size, 1)), dim=1)

        x1_t, _ = self.path(t)
        hat_x1_t = cat((x1_t, t * ones(self.batch_size, 1)), dim=1)

        return hat_x1_t, self.approx_x2(t)

    def reset(self):
        self.x2_parent_wrt_time = {0.: zeros(self.batch_size, self.n - 1)}
        self.path.reset()
