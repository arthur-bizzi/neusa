cat << 'EOF' > /mnt/user-data/outputs/wavelet_basis_1d.py
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
    """
    1D Daubechies DB_{2p} wavelet basis for spectral differentiation.
    Uses Beylkin's connection coefficients method (periodic domain).
    Pure PyTorch implementation.

    Supports DB2 (p=2), DB4 (p=4), DB6 (p=6).

    The scaling function phi has support [0, 2p-1], so the connection
    coefficients Gamma_n = int phi(x) phi'(x-n) dx are nonzero exactly for
    n in [-(2p-2), 2p-2]  (symmetric, size 4p-3).

    At scale J with M = 2^J collocation points, D acts on nodal values
    directly (Beylkin 1992, periodic case).

    Interface:
        dbt   : physical values -> wavelet coefficients  u -> c
        idbt  : wavelet coefficients -> physical values  c -> u
        diff  : first derivative  u -> du/dx   shape (..., M) -> (..., M)
        diff2 : second derivative u -> d²u/dx²
    """

    def __init__(self, p, J, domain, device="cpu", cascade_iters=12):
        """
        Parameters
        ----------
        p             : int    — Daubechies order (2, 4, or 6)
        J             : int    — scale level, M = 2^J collocation points
        domain        : (a,b)  — physical domain (periodic)
        device        : str    — torch device
        cascade_iters : int    — iterations for cascade evaluation of phi (dbt/idbt)
        """
        assert p in _DB_FILTERS, f"p must be in {list(_DB_FILTERS.keys())}, got {p}"

        self.p = p
        self.J = J
        self.M = 2 ** J
        self.device = device
        self.a, self.b = domain
        self.L = self.b - self.a
        self.cascade_iters = cascade_iters

        # Filter coefficients
        self.h = torch.tensor(_DB_FILTERS[p], dtype=torch.float64)

        # Support of connection coefficients
        self.n_min = -(2 * p - 2)
        self.n_max =  (2 * p - 2)

        # Collocation points on [a, b)
        self.x = torch.linspace(self.a, self.b, self.M + 1)[:-1]

        # Chain rule: d/dx = (2^J / L) * d/dz
        self.jacobian = (2 ** J) / self.L

        # Connection coefficients Gamma_n
        self.gamma = self._compute_connection_coefficients()

        # Differentiation matrices (act on nodal values directly)
        self.D  = self._build_diff_matrix(order=1)
        self.D2 = self._build_diff_matrix(order=2)

        # Evaluation matrix for dbt/idbt (via cascade)
        self.Wx     = self._build_Wx()
        self.Wx_inv = torch.linalg.inv(self.Wx)

        self.to(device)


    def _cascade(self):
        """
        Evaluate phi on a fine grid of step 2^{-cascade_iters} via the
        subdivision algorithm:
            phi^{n+1}(x) = sqrt(2) * sum_k h_k phi^n(2x - k)
        Returns phi values on grid of size (len(h)-1) * 2^n_iter + 1.
        """
        h = self.h
        nf = len(h)
        support = nf - 1          # phi support = [0, nf-1]
        fine = 2 ** self.cascade_iters
        grid_size = support * fine + 1

        # Init: box function on [0, 1)
        phi = torch.zeros(grid_size, dtype=torch.float64)
        phi[:fine] = 1.0

        for _ in range(self.cascade_iters):
            phi_new = torch.zeros(grid_size, dtype=torch.float64)
            for k in range(nf):
                hk = h[k].item()
                for i in range(grid_size):
                    src = 2 * i - k * fine
                    if 0 <= src < grid_size:
                        phi_new[i] += math.sqrt(2) * hk * phi[src]
            phi = phi_new

        # Normalize so integral = 1
        dx = 1.0 / fine
        phi = phi / (phi.sum() * dx)

        return phi, fine


    def _build_Wx(self):
        """
        Wx[j,k] = phi(j - k mod M) evaluated via cascade.
        Collocation points are at integer positions z_j = j (scale J).
        """
        M = self.M
        phi, fine = self._cascade()
        grid_size = len(phi)

        Wx = torch.zeros((M, M), dtype=torch.float64)
        for j in range(M):
            for k in range(M):
                arg = (j - k) % M
                fi = int(arg * fine)
                if 0 <= fi < grid_size:
                    Wx[j, k] = phi[fi]
        return Wx

    def _compute_connection_coefficients(self):
        h = self.h
        n_min, n_max = self.n_min, self.n_max
        nf = len(h)
        size = n_max - n_min + 1

        def idx(n): return n - n_min

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

    def dbt(self, u):
        """
        Direct Basis Transform: physical values -> wavelet coefficients.
        u     : (..., M) tensor
        return: (..., M) tensor of coefficients c
        """
        return u @ self.Wx_inv

    def idbt(self, c):
        """
        Inverse Basis Transform: wavelet coefficients -> physical values.
        c     : (..., M) tensor of coefficients
        return: (..., M) tensor of nodal values
        """
        return c @ self.Wx

    def diff(self, u):
        """
        First derivative of u (acts on nodal values directly).
        u : (..., M) tensor
        """
        return u @ self.D.T

    def diff2(self, u):
        """
        Second derivative of u (acts on nodal values directly).
        u : (..., M) tensor
        """
        return u @ self.D2.T

    def to(self, device):
        self.x      = self.x.to(device)
        self.h      = self.h.to(device)
        self.D      = self.D.to(device)
        self.D2     = self.D2.to(device)
        self.Wx     = self.Wx.to(device)
        self.Wx_inv = self.Wx_inv.to(device)
        self.device = device
        return self

