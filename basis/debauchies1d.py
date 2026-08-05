import math
import torch


_DB_FILTERS = {
    2: [
        0.4829629131445341,
        0.8365163037378077,
        0.2241438680420134,
       -0.1294095225512603,
    ],
    4: [
        0.2303778133088552,
        0.7148465705525415,
        0.6308807679295904,
       -0.0279837694169839,
       -0.1870348117188811,
        0.0308413818359867,
        0.0328830116668852,
       -0.0105974017850690,
    ],
    6: [
       -0.0010773010849956,
        0.0047772575109455,
        0.0005538422011615,
       -0.0315820393174860,
        0.0275228655303060,
        0.0975016055870790,
       -0.1297668675670950,
       -0.2262646939651690,
        0.3152503517091980,
        0.7511339080215780,
        0.4946238903984570,
        0.1115407433500810,
    ],
}


class WaveletBasis1D:

    def __init__(self, p, J, domain, device="cpu"):

        assert p in _DB_FILTERS, f"p must be in {list(_DB_FILTERS.keys())}, got {p}"

        self.p = p
        self.J = J
        self.M = 2 ** J
        self.device = device
        self.a, self.b = domain
        self.L = self.b - self.a

        self.h = torch.tensor(_DB_FILTERS[p], dtype=torch.float64)

        self.n_min = -(2 * p - 2)
        self.n_max = 2 * p - 2

        self.x = torch.linspace(self.a, self.b, self.M + 1)[:-1]

        self.jacobian = (2 ** J) / self.L

        # Connection coefficients Gamma_n
        self.gamma = self._compute_connection_coefficients()

        # Differentiation matrices (act on nodal values directly)
        self.D  = self._build_diff_matrix(order=1)
        self.D2 = self._build_diff_matrix(order=2)

        self.to(device)


    def _compute_connection_coefficients(self):
        h = self.h
        n_min, n_max = self.n_min, self.n_max
        nf = len(h)
        size = n_max - n_min + 1

        def idx(n): return n - n_min

        # Build matrix A such that A @ gamma = gamma
        A = torch.zeros((size, size), dtype=torch.float64)
        for n in range(n_min, n_max + 1):
            for l in range(nf):
                for m in range(nf):
                    rhs_n = 2 * n + l - m
                    if n_min <= rhs_n <= n_max:
                        A[idx(n), idx(rhs_n)] += 2 * h[l] * h[m]

        eigen_eq = A - torch.eye(size, dtype=torch.float64)

        norm_row = torch.tensor(
            [float(n) for n in range(n_min, n_max + 1)],
            dtype=torch.float64
        ).unsqueeze(0)

        system = torch.vstack([eigen_eq, norm_row])
        rhs = torch.zeros(size + 1, dtype=torch.float64)
        rhs[-1] = -1.0

        result = torch.linalg.lstsq(system, rhs.unsqueeze(1))
        gamma_vals = result.solution.squeeze()

        return {n: gamma_vals[idx(n)].item() for n in range(n_min, n_max + 1)}


    def _build_diff_matrix(self, order=1):
        M = self.M
        gamma = self.gamma

        D = torch.zeros((M, M), dtype=torch.float64)
        for j in range(M):
            for k in range(M):
                n = (j - k) % M
                if n > M // 2:
                    n -= M
                if n in gamma:
                    D[j, k] = gamma[n]
        D = D * self.jacobian

        if order == 1:
            return D
        else:
            return D @ D


    def diff(self, u):
        return u @ self.D.T

    def diff2(self, u):
        return u @ self.D2.T

    def to(self, device):
        self.x   = self.x.to(device)
        self.h   = self.h.to(device)
        self.D   = self.D.to(device)
        self.D2  = self.D2.to(device)
        self.device = device
        return self


