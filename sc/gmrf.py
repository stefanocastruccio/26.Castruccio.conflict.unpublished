"""Latent Gaussian Markov random field blocks.

Each block represents one latent effect x_k ~ N(0, Q_k(theta_k)^{-1}) and knows
  * its precision Q(theta),
  * a square factor R(theta) with R^T R = Q (used for sampling),
  * log|Q(theta)| (computed cheaply from its structure),
  * a log-prior for its hyperparameters theta (on the internal, unconstrained scale).

The spatial block is a finite-volume discretisation of the Matern SPDE
(Lindgren, Rue & Lindstrom 2011, alpha = 2, nu = 1) on a regular lon/lat grid
on the sphere, so distances, areas and the latitude-dependent shrinking of
east-west spacing are handled in km. Cells may be masked (e.g. ocean), and the
grid may wrap around in longitude for global data.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

EARTH_RADIUS_KM = 6371.0


def sparse_logdet(M: sp.spmatrix) -> float:
    """log|M| for a sparse symmetric positive definite matrix (via sparse LU)."""
    lu = splu(sp.csc_matrix(M), permc_spec="MMD_AT_PLUS_A",
              diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    return float(np.sum(np.log(np.abs(lu.U.diagonal()))))


# --------------------------------------------------------------------------
# Lattice geometry on the sphere
# --------------------------------------------------------------------------
class SphereLattice:
    """Regular lon/lat grid (cell centres) with an optional mask.

    Builds the lumped mass matrix C (cell areas, km^2) and the finite-volume
    stiffness matrix G (sum over shared edges of edge_length / centre_distance),
    so that  -Laplace_S2  ~=  C^{-1} G.
    """

    def __init__(self, lon_centres, lat_centres, mask=None, periodic_lon=False):
        self.lon_c = np.asarray(lon_centres, float)
        self.lat_c = np.asarray(lat_centres, float)
        nx, ny = len(self.lon_c), len(self.lat_c)
        dlon = np.deg2rad(abs(self.lon_c[1] - self.lon_c[0]))
        dlat = np.deg2rad(abs(self.lat_c[1] - self.lat_c[0]))
        if mask is None:
            mask = np.ones((ny, nx), bool)
        self.mask = np.asarray(mask, bool)
        idx = -np.ones((ny, nx), int)
        idx[self.mask] = np.arange(self.mask.sum())
        self.index = idx
        self.n = int(self.mask.sum())
        jj, ii = np.nonzero(self.mask)  # row (lat), col (lon)
        self.lat = self.lat_c[jj]
        self.lon = self.lon_c[ii]
        self.row, self.col = jj, ii
        R = EARTH_RADIUS_KM
        phi = np.deg2rad(self.lat)
        self.area = R ** 2 * dlon * (np.sin(phi + dlat / 2) - np.sin(phi - dlat / 2))

        rows, cols, vals = [], [], []
        # east-west neighbours: edge length R*dlat, centre distance R*cos(phi)*dlon
        for j in range(ny):
            cphi = np.cos(np.deg2rad(self.lat_c[j]))
            w = (R * dlat) / (R * cphi * dlon)
            rng_i = range(nx) if periodic_lon else range(nx - 1)
            for i in rng_i:
                a, b = idx[j, i], idx[j, (i + 1) % nx]
                if a >= 0 and b >= 0 and a != b:
                    rows += [a, b]; cols += [b, a]; vals += [w, w]
        # north-south neighbours: edge length R*cos(phi_edge)*dlon, distance R*dlat
        for j in range(ny - 1):
            phi_e = np.deg2rad(0.5 * (self.lat_c[j] + self.lat_c[j + 1]))
            w = (R * np.cos(phi_e) * dlon) / (R * dlat)
            for i in range(nx):
                a, b = idx[j, i], idx[j + 1, i]
                if a >= 0 and b >= 0:
                    rows += [a, b]; cols += [b, a]; vals += [w, w]
        W = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        self.adjacency = (W > 0).astype(float).tocsr()
        self.G = (sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsc()
        self.C = sp.diags(self.area).tocsc()

    def smooth_basis(self, k: int):
        """k smoothest eigenvectors of the generalized problem G v = lambda C v
        (the constant vector first). Used e.g. for spatial+ residualisation."""
        from scipy.sparse.linalg import eigsh
        vals, vecs = eigsh(self.G, k=k, M=self.C, sigma=-1e-8, which="LM")
        order = np.argsort(vals)
        return vecs[:, order]


# --------------------------------------------------------------------------
# Hyperprior helpers (all on the log scale of the parameter, incl. Jacobian)
# --------------------------------------------------------------------------
def log_pc_sd(log_sigma, u, alpha):
    """PC prior on a standard deviation: P(sigma > u) = alpha."""
    lam = -np.log(alpha) / u
    s = np.exp(log_sigma)
    return np.log(lam) - lam * s + log_sigma


def log_pc_range2d(log_range, u, alpha):
    """PC prior on a 2-d Matern range (Fuglstad et al. 2019): P(range < u) = alpha."""
    lam = -np.log(alpha) * u
    r = np.exp(log_range)
    return np.log(lam) - 2 * log_range - lam / r + log_range


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------
class Block:
    name: str
    size: int
    n_theta: int = 0

    def theta0(self):
        return np.zeros(self.n_theta)

    def Q(self, th):
        R = self.R(th)
        return (R.T @ R).tocsc()

    def R(self, th):
        raise NotImplementedError

    def logdet(self, th):
        raise NotImplementedError

    def log_prior(self, th):
        return 0.0

    def bounds(self):
        """Box constraints on the internal hyperparameters for the outer optimiser."""
        return [(-8.0, 10.0)] * self.n_theta

    def describe(self, th):
        return {}


# plausible ranges on the internal scale: log sd in [log 0.005, log 20],
# log range in [log 10 km, log 20,000 km], atanh(rho) in [-3, 3] (|rho| <= 0.995)
LOG_SD = (np.log(0.005), np.log(20.0))
LOG_RANGE = (np.log(10.0), np.log(2.0e4))
ATANH_RHO = (-3.0, 3.0)


class FixedBlock(Block):
    """Fixed effects with a vague N(0, 1/prec) prior (integrated out by Laplace)."""

    def __init__(self, p, names=None, prec=1e-4, name="fixed"):
        self.size, self.prec, self.name = p, prec, name
        self.names = names or [f"b{i}" for i in range(p)]

    def R(self, th):
        return sp.identity(self.size, format="csc") * np.sqrt(self.prec)

    def logdet(self, th):
        return self.size * np.log(self.prec)


class IIDBlock(Block):
    """Exchangeable random effect, e.g. country intercepts. theta = [log sd]."""
    n_theta = 1

    def __init__(self, n, name="iid", sd_init=0.5, pc_u=1.0, pc_alpha=0.01):
        self.size, self.name = n, name
        self._t0 = np.array([np.log(sd_init)])
        self.pc = (pc_u, pc_alpha)

    def theta0(self):
        return self._t0.copy()

    def R(self, th):
        return sp.identity(self.size, format="csc") / np.exp(th[0])

    def logdet(self, th):
        return -2 * self.size * th[0]

    def log_prior(self, th):
        return log_pc_sd(th[0], *self.pc)

    def bounds(self):
        return [LOG_SD]

    def describe(self, th):
        return {f"{self.name}.sd": float(np.exp(th[0]))}


class MaternBlock(Block):
    """Matern (nu = 1) field on a SphereLattice via the SPDE / GMRF representation.

    theta = [log range_km, log sd]. Written in the form of Bakka et al. (2019),

        u - div( r(s)^2 / 8  grad u ) = r(s) sqrt(pi/2) sd W,

    which for a constant range r is the Matern SPDE with kappa = sqrt(8)/r and
    marginal variance sd^2. Finite volumes give  K u = D eps  with
    K = C + (r^2/8) G_m,  D = diag(r_i sqrt(pi/2) sd sqrt(a_i)),  Q = K D^{-2} K.

    `barrier` (boolean per cell) marks cells (e.g. sea) whose range is
    `barrier_frac` times the land range: the barrier model. Such cells carry no
    data; they remove the variance inflation of a reflecting coastline while
    preventing correlation from leaking across straits. The diffusion coefficient
    on an edge between cells i and j is the harmonic mean of (r_i/r)^2 and (r_j/r)^2,
    the usual two-point flux choice for a discontinuous coefficient (it equals the
    arithmetic mean when neither cell is a barrier).
    """
    n_theta = 2

    def __init__(self, lattice: SphereLattice, name="field", range_init=500.0,
                 sd_init=0.5, pc_range=(100.0, 0.05), pc_sd=(1.0, 0.01),
                 barrier=None, barrier_frac=0.5):
        self.lat, self.name, self.size = lattice, name, lattice.n
        self._t0 = np.log([range_init, sd_init])
        self.pc_range, self.pc_sd = pc_range, pc_sd
        m = np.ones(lattice.n)                       # (r_i / r)^2
        if barrier is not None:
            m[np.asarray(barrier, bool)] = barrier_frac ** 2
        self._m = m
        W = sp.diags(lattice.G.diagonal()) - lattice.G          # edge weights l/d
        W = sp.coo_matrix(W)
        mij = 2.0 * m[W.row] * m[W.col] / (m[W.row] + m[W.col])   # harmonic mean
        Wm = sp.csr_matrix((W.data * mij, (W.row, W.col)), shape=W.shape)
        self._Gm = (sp.diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm).tocsc()
        self._dfac = np.sqrt(m * lattice.area * np.pi / 2)      # D / (r sd)

    def theta0(self):
        return self._t0.copy()

    def K(self, th):
        rng = np.exp(th[0])
        return (self.lat.C + (rng ** 2 / 8.0) * self._Gm).tocsc()

    def _d(self, th):
        rng, sd = np.exp(th)
        return rng * sd * self._dfac

    def R(self, th):
        return (sp.diags(1.0 / self._d(th)) @ self.K(th)).tocsc()

    def logdet(self, th):
        return 2 * sparse_logdet(self.K(th)) - 2 * float(np.sum(np.log(self._d(th))))

    def log_prior(self, th):
        return log_pc_range2d(th[0], *self.pc_range) + log_pc_sd(th[1], *self.pc_sd)

    def bounds(self):
        return [LOG_RANGE, LOG_SD]

    def describe(self, th):
        rng, sd = np.exp(th)
        return {f"{self.name}.range_km": float(rng), f"{self.name}.sd": float(sd)}

    def sample(self, th, rng: np.random.Generator, n=1):
        """Exact draw(s) from N(0, Q^{-1})."""
        lu = splu(self.K(th))
        e = rng.standard_normal((self.size, n)) * self._d(th)[:, None]
        return lu.solve(e)


class SpaceTimeAR1Block(Block):
    """Separable space-time field: AR(1) in time (unit marginal variance)
    Kronecker a Matern field in space. Latent index = t * n_space + s.
    theta = [log range_km, log sd, logit(rho)]  (rho in (-1, 1) via tanh-like map)."""
    n_theta = 3

    def __init__(self, lattice: SphereLattice, T: int, name="st", range_init=300.0,
                 sd_init=0.3, rho_init=0.7, pc_range=(100.0, 0.05), pc_sd=(1.0, 0.01),
                 barrier=None, barrier_frac=0.5):
        self.space = MaternBlock(lattice, name=name, range_init=range_init,
                                 sd_init=sd_init, pc_range=pc_range, pc_sd=pc_sd,
                                 barrier=barrier, barrier_frac=barrier_frac)
        self.T, self.name = T, name
        self.size = lattice.n * T
        self._t0 = np.r_[self.space.theta0(), np.arctanh(rho_init)]

    def theta0(self):
        return self._t0.copy()

    def _Rt(self, rho):
        T = self.T
        s = np.sqrt(1 - rho ** 2)
        main = np.r_[1.0, np.full(T - 1, 1.0 / s)]
        off = np.full(T - 1, -rho / s)
        return sp.diags([main, off], [0, -1], shape=(T, T), format="csc")

    def R(self, th):
        rho = np.tanh(th[2])
        return sp.kron(self._Rt(rho), self.space.R(th[:2]), format="csc")

    def logdet(self, th):
        rho = np.tanh(th[2])
        ld_t = -(self.T - 1) * np.log(1 - rho ** 2)
        return self.space.size * ld_t + self.T * self.space.logdet(th[:2])

    def log_prior(self, th):
        # weak N(0, 1.5^2) on atanh(rho)
        return self.space.log_prior(th[:2]) - 0.5 * (th[2] / 1.5) ** 2

    def bounds(self):
        return [LOG_RANGE, LOG_SD, ATANH_RHO]

    def describe(self, th):
        d = self.space.describe(th[:2])
        d[f"{self.name}.rho"] = float(np.tanh(th[2]))
        return d


class CopyBlock(Block):
    """Shared latent component entering two linear predictors with a loading.

    Latent vector [a; b] with a ~ base GMRF (e.g. MaternBlock or SpaceTimeAR1Block)
    and b = lam * a + e, e ~ N(0, I / tau) with a large fixed tau (the "copy"
    construction of INLA). The design maps outcome-1 rows to a and outcome-2 rows
    to b, so outcome 2 receives lam times the shared component. The joint precision is

        Q = [[Q_a + lam^2 tau I, -lam tau I], [-lam tau I, tau I]] = R'R,
        R = [[R_a, 0], [-lam sqrt(tau) I, sqrt(tau) I]],

    with log|Q| = log|Q_a| + m log tau, so lam does not enter the determinant.
    theta = [theta_base..., lam]; lam has a N(0, 2^2) prior and is bounded to [-5, 5].
    """

    def __init__(self, base: Block, name="shared", tau=1e4, lam_init=1.0):
        self.base, self.name, self.tau = base, name, float(tau)
        self.m = base.size
        self.size = 2 * base.size
        self.n_theta = base.n_theta + 1
        self._lam0 = float(lam_init)

    def theta0(self):
        return np.r_[self.base.theta0(), self._lam0]

    def R(self, th):
        Ra = self.base.R(th[:-1])
        lam, st = th[-1], np.sqrt(self.tau)
        I = sp.identity(self.m, format="csc")
        return sp.bmat([[Ra, None], [-lam * st * I, st * I]], format="csc")

    def logdet(self, th):
        return self.base.logdet(th[:-1]) + self.m * np.log(self.tau)

    def log_prior(self, th):
        return self.base.log_prior(th[:-1]) - 0.5 * (th[-1] / 2.0) ** 2

    def bounds(self):
        return list(self.base.bounds()) + [(-5.0, 5.0)]

    def describe(self, th):
        d = {k.replace(self.base.name, self.name, 1): v for k, v in self.base.describe(th[:-1]).items()}
        d[f"{self.name}.loading"] = float(th[-1])
        return d
