from RoughPaths import *
from math import isclose


class StratonovichBrownianRoughPath(RoughPath):
    def __init__(self, n, batch_size=1, device='cpu'):
        super(StratonovichBrownianRoughPath, self).__init__(n, batch_size, device=device)
        self.vals_bm1 = {0: zeros(batch_size, n, device=device)}
        self.vals_bm2 = {0: zeros(batch_size, n, n, device=device)}
        self.last_bm2_t = 0
        self.last_bm2_val = self.vals_bm2[0]

    p = 2.00001

    def _bm(self, t):
        assert t >= 0, f"Only positive time possible, but got {t}"
        if t in self.vals_bm1.keys():
            return self.vals_bm1[t]

        max_t = max(self.vals_bm1.keys())
        if t > max_t:
            last_t = max_t
            Z = normal(0, ones(self.batch_size, self.n, device=self.device))
            B_t = self.vals_bm1[last_t] + sqrt(t - last_t) * Z
            self.vals_bm1[t] = B_t
            return B_t

        last_t = max([s for s in self.vals_bm1.keys() if s < t])
        if isclose(last_t, t):
            return self.vals_bm1[last_t]
        next_t = min([s for s in self.vals_bm1.keys() if s > t])
        # is the difference only a floating point rounding error?
        if isclose(next_t, t):
            return self.vals_bm1[next_t]

        # Brownian bridge construction
        Z = normal(0, ones(self.batch_size, self.n, device=self.device))
        B_t = (t - last_t) / (next_t - last_t) * self.vals_bm1[next_t] + (next_t - t) / (next_t - last_t) \
              * self.vals_bm1[last_t] + sqrt((next_t - t) * (t - last_t) / (next_t - last_t)) * Z
        self.vals_bm1[t] = B_t

        if self.n > 1:
            bm1_keys = sorted(set(self.vals_bm1.keys()))
            for k in bm1_keys:
                if k < t:
                    self.last_bm2_t = k
                    self.last_bm2_val = self.vals_bm2[k]
                    continue
                if k > t:
                    # use all sampled points for approximation of iterated integrals
                    self.vals_bm2.pop(k)
                self._bm2(k)
        return B_t

    def _bm2(self, t):
        if self.n == 1:
            return self._bm(t).square().view(self.batch_size, 1, 1) / 2

        assert t in self.vals_bm1.keys(), f"Iterated integrals can only be calculated after value at this point is " \
                                          f"called; but got t={t}"
        if t in self.vals_bm2.keys():
            return self.vals_bm2[t]

        assert t > self.last_bm2_t, f"Only calculate forward; no inbetween steps"

        bm2_t = self.last_bm2_val + einsum('bi, bj -> bij', self.vals_bm1[t], self.vals_bm1[t] - self.vals_bm1[self.last_bm2_t])
        self.vals_bm2[t] = bm2_t
        self.last_bm2_t = t
        self.last_bm2_val = bm2_t

        return bm2_t

    def reset(self):
        super(StratonovichBrownianRoughPath, self).reset()
        self.vals_bm1 = {0: zeros(self.batch_size, self.n, device=self.device)}
        self.vals_bm2 = {0: zeros(self.batch_size, self.n, self.n, device=self.device)}
        self.last_bm2_t = 0
        self.last_bm2_val = self.vals_bm2[0]

    def __call__(self, t):
        super(StratonovichBrownianRoughPath, self).__call__(t)
        B_t = self._bm(t)
        B2_t = self._bm2(t)
        if self.n > 1:
            for i in range(self.n):
                B2_t[:, i, i] = B_t[:, i].square() / 2
        return B_t, B2_t


class ItoBrownianRoughPath(StratonovichBrownianRoughPath):
    def __call__(self, t):
        B_t, B2_t = super(ItoBrownianRoughPath, self).__call__(t)
        diff_tens = t / 2 * ones(self.batch_size, device=self.device)
        for i in range(self.n):
            B2_t[:, i, i] -= diff_tens
        return B_t, B2_t


class StratonovichFractionalBrownianRoughPath(StratonovichBrownianRoughPath):
    def __init__(self, n, H, steps=1000, T=1., batch_size=1, device='cpu'):
        super(StratonovichFractionalBrownianRoughPath, self).__init__(n, batch_size, device=device)
        assert H > 1/3, f"As we only consider rough paths for p < 3, we can only deal with H > 1/3; but got {H} instead."
        self.H = H
        self.steps = steps
        self.T = T
        self.del_t = self.T / self.steps

        self.vals_bm = None
        self.std_div_mat = None
        self._precompute_vals()

    def cov(self, s, t):
        return (pow(s, 2*self.H) + pow(t, 2*self.H) - pow(abs(t - s), 2*self.H))/2

    def _precompute_vals(self):
        if self.std_div_mat is None:
            cov_mat = [[self.cov(self.del_t * (i + 1), self.del_t * (j + 1)) for i in range(self.steps)] for j in range(self.steps)]
            cov_mat = tensor(cov_mat, device=self.device)
            self.std_div_mat = cholesky(cov_mat)

        z = normal(0, ones(self.batch_size, self.n, self.steps, device=self.device))
        self.vals_bm = einsum('ij, bnj -> bni', self.std_div_mat, z)

    def _bm(self, t):
        assert t >= 0, f"Only positive time possible, but got {t}"
        step = t / self.del_t
        step_lower = int(step) - 1
        step_upper = min([step_lower + 1, self.steps - 1])
        step_perc = step - step_lower
        if step_lower < 0:
            val_lower = zeros(self.batch_size, self.n, device=self.device)
        else:
            val_lower = self.vals_bm[:, :, step_lower]
        val_upper = self.vals_bm[:, :, step_upper]
        return (1 - step_perc) * val_lower + step_perc * val_upper

    def reset(self):
        super(StratonovichFractionalBrownianRoughPath, self).reset()
        self.vals_bm = None
        self._precompute_vals()


class ItoFractionalBrownianRoughPath(StratonovichFractionalBrownianRoughPath):
    def __call__(self, t):
        x1, x2 = super(ItoFractionalBrownianRoughPath, self).__call__(t)
        diff_tens = pow(t, 2 * self.H) / 2 * ones(self.batch_size, device=self.device)
        for i in range(self.n):
            x2[:, i, i] -= diff_tens
        return x1, x2
