from RoughPaths import *


class StratonovichBrownianRoughPath(RoughPath):
    def __init__(self, n, batch_size=1):
        super(StratonovichBrownianRoughPath, self).__init__(n, batch_size)
        self.vals_bm1 = {0: zeros(batch_size, n)}
        self.vals_bm2 = {0: zeros(batch_size, n, n)}

    p = 2.00001

    def _bm(self, t):
        assert t >= 0, f"Only positive time possible, but got {t}"
        if t in self.vals_bm1.keys():
            return self.vals_bm1[t]

        max_t = max(self.vals_bm1.keys())
        if t > max_t:
            last_t = max_t
            Z = normal(0, ones(self.batch_size, self.n))
            B_t = self.vals_bm1[last_t] + sqrt(t - last_t) * Z
            self.vals_bm1[t] = B_t
            return B_t

        # Brownian bridge construction
        last_t = max([s for s in self.vals_bm1.keys() if s < t])
        next_t = min([s for s in self.vals_bm1.keys() if s > t])
        Z = normal(0, ones(self.batch_size, self.n))
        B_t = (t - last_t) / (next_t - last_t) * self.vals_bm1[next_t] + (next_t - t) / (next_t - last_t) \
              * self.vals_bm1[last_t] + sqrt((next_t - t) * (t - last_t) / (next_t - last_t)) * Z
        self.vals_bm1[t] = B_t
        for k in self.vals_bm2.keys():
            if k < t:
                continue
            # use all sampled points for approximation of iterated integrals
            self.vals_bm2.pop(k)
        return B_t

    def _bm2(self, t):
        assert t in self.vals_bm1.keys(), f"Iterated integrals can only be calculated after value at this point is " \
                                          f"called; but got t={t}"
        if t in self.vals_bm2.keys():
            return self.vals_bm2[t]

        t_start = max(self.vals_bm2.keys())
        bm2_s = self.vals_bm2[t_start]
        s_last = t_start
        for s in sorted(self.vals_bm1.keys()):
            if s <= t_start:
                continue

            bm2_s = bm2_s + einsum('bi, bj -> bij', self.vals_bm1[s], self.vals_bm1[s] - self.vals_bm1[s_last])
            self.vals_bm2[s] = bm2_s

            if s == t:
                break
        return bm2_s

    def reset(self):
        super(StratonovichBrownianRoughPath, self).reset()
        self.vals_bm1 = {0: zeros(self.batch_size, self.n)}
        self.vals_bm2 = {0: zeros(self.batch_size, self.n, self.n)}

    def __call__(self, t):
        super(StratonovichBrownianRoughPath, self).__call__(t)
        B_t = self._bm(t)
        B2_t = self._bm2(t)
        for i in range(self.n):
            B2_t[:, i, i] = 1 / 2 * B_t[:, i].square()
        return B_t, B2_t


class ItoBrownianRoughPath(StratonovichBrownianRoughPath):
    def __call__(self, t):
        B_t, B2_t = super(ItoBrownianRoughPath, self).__call__(t)
        return B_t, B2_t - t / 2 * eye(self.n).view(1, self.n, self.n).repeat(self.batch_size, 1, 1)


class StratonovichFractionalBrownianRoughPath(StratonovichBrownianRoughPath):
    def __init__(self, n, H, steps=1000, T=1., batch_size=1):
        super(StratonovichFractionalBrownianRoughPath, self).__init__(n, batch_size)
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
            cov_mat = tensor(cov_mat)
            self.std_div_mat = cholesky(cov_mat)

        z = normal(0, ones(self.batch_size, self.n, self.steps))
        self.vals_bm = einsum('ij, bnj -> bni', self.std_div_mat, z)

    def _bm(self, t):
        assert t >= 0, f"Only positive time possible, but got {t}"
        step = t / self.del_t
        step_lower = int(step) - 1
        step_upper = min([step_lower + 1, self.steps - 1])
        step_perc = step - step_lower
        if step_lower < 0:
            val_lower = zeros(self.batch_size, self.n)
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
        return x1, x2 - pow(t, 2 * self.H) / 2 * eye(self.n).view(1, self.n, self.n).repeat(self.batch_size, 1, 1)
