"""Laplace-approximation latent Gaussian model (empirical-Bayes INLA / TMB style).

Model:   y_i ~ F(mu_i),  g(mu_i) = offset_i + (A x)_i,
         x = (x_1, ..., x_K),  x_k ~ N(0, Q_k(theta_k)^{-1}) independently.

For given theta the posterior mode x* is found by damped Newton; the marginal
likelihood is approximated by

  log p(y | theta) ~= log p(y | x*) - 1/2 x*'Q x* + 1/2 log|Q| - 1/2 log|Q + H|,

with H = A' W A the observed information of the likelihood at x*. theta is set to
the posterior mode of  log p(y | theta) + log p(theta)  (L-BFGS-B), and the
latent posterior is the Gaussian N(x*, (Q + H)^{-1}).

Families: 'bernoulli' / 'binomial' (logit link), 'poisson' (log link).
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from scipy.special import expit, gammaln

from .gmrf import Block


def _lu(M):
    return splu(sp.csc_matrix(M), permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))


def nested_dissection(M):
    """Fill-reducing METIS nested-dissection ordering of a symmetric sparse
    matrix (much better than minimum degree for space-time lattices).
    Returns None if pymetis is not installed."""
    try:
        import pymetis
    except ImportError:
        return None
    G = sp.csr_matrix(M, copy=True)
    G.setdiag(0); G.eliminate_zeros()
    perm, _ = pymetis.nested_dissection(pymetis.CSRAdjacency(G.indptr, G.indices))
    return np.asarray(perm)


class Factor:
    """Sparse LU of an SPD matrix with an optional fixed symmetric permutation."""

    def __init__(self, M, perm=None):
        self.perm = perm
        if perm is None:
            self.lu = _lu(M)
        else:
            Mp = sp.csr_matrix(M)[perm][:, perm]
            self.lu = splu(sp.csc_matrix(Mp), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                           options=dict(SymmetricMode=True))

    def solve(self, b):
        if self.perm is None:
            return self.lu.solve(b)
        x = np.empty_like(b, dtype=float)
        x[self.perm] = self.lu.solve(np.ascontiguousarray(b[self.perm]))
        return x

    def logdet(self):
        return float(np.sum(np.log(np.abs(self.lu.U.diagonal()))))


class LatentGaussianModel:
    def __init__(self, y, blocks: list[Block], designs: list, offset=None,
                 family="bernoulli", trials=None):
        self.y = np.asarray(y, float)
        n = len(self.y)
        self.blocks = blocks
        self.designs = [sp.csr_matrix(D) for D in designs]
        for b, D in zip(blocks, self.designs):
            assert D.shape == (n, b.size), (b.name, D.shape, b.size)
        self.A = sp.hstack(self.designs, format="csr")
        self.At = self.A.T.tocsr()
        self.offset = np.zeros(n) if offset is None else np.asarray(offset, float)
        self.family = family
        self.trials = np.ones(n) if trials is None else np.asarray(trials, float)
        self.slices, start = [], 0
        for b in blocks:
            self.slices.append(slice(start, start + b.size)); start += b.size
        self.dim = start
        self.tslices, start = [], 0
        for b in blocks:
            self.tslices.append(slice(start, start + b.n_theta)); start += b.n_theta
        self.n_theta = start
        self._x = np.zeros(self.dim)
        self.n_evals = 0
        self._perm = False     # computed lazily: the sparsity of Q + H does not change

    # ---- likelihood pieces ------------------------------------------------
    def _loglik(self, eta):
        y, m = self.y, self.trials
        if self.family in ("bernoulli", "binomial"):
            return float(np.sum(y * eta - m * np.logaddexp(0, eta)))
        if self.family == "poisson":
            return float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1)))
        raise ValueError(self.family)

    def _grad_w(self, eta):
        if self.family in ("bernoulli", "binomial"):
            p = expit(eta)
            return self.y - self.trials * p, self.trials * p * (1 - p)
        mu = np.exp(eta)
        return self.y - mu, mu

    # ---- prior pieces -------------------------------------------------------
    def _split(self, theta):
        return [theta[s] for s in self.tslices]

    def Q(self, theta):
        return sp.block_diag([b.Q(t) for b, t in zip(self.blocks, self._split(theta))],
                             format="csc")

    def R(self, theta):
        return sp.block_diag([b.R(t) for b, t in zip(self.blocks, self._split(theta))],
                             format="csc")

    def logdet_Q(self, theta):
        return sum(b.logdet(t) for b, t in zip(self.blocks, self._split(theta)))

    def log_prior(self, theta):
        return sum(b.log_prior(t) for b, t in zip(self.blocks, self._split(theta)))

    # ---- inner optimisation -------------------------------------------------
    def mode(self, theta, x0=None, tol=1e-8, maxit=50):
        Q = self.Q(theta)
        x = self._x.copy() if x0 is None else x0.copy()

        def obj(x):
            eta = self.offset + self.A @ x
            return self._loglik(eta) - 0.5 * x @ (Q @ x), eta

        f, eta = obj(x)
        if self._perm is False:
            _, w = self._grad_w(eta)
            self._perm = (nested_dissection(Q + self.At @ sp.diags(w) @ self.A)
                          if self.dim > 2000 else None)
        self._converged = False
        for it in range(maxit):
            g_lik, w = self._grad_w(eta)
            grad = self.At @ g_lik - Q @ x
            lu = Factor(Q + self.At @ sp.diags(w) @ self.A, self._perm)
            step = lu.solve(grad)
            t = 1.0
            while True:
                fn, etan = obj(x + t * step)
                if fn >= f - 1e-10 or t < 1e-6:
                    break
                t *= 0.5
            x, df = x + t * step, fn - f
            f, eta = fn, etan
            if abs(df) < tol * (1 + abs(f)) and np.max(np.abs(t * step)) < 1e-5:
                self._converged = True
                break
        _, w = self._grad_w(eta)
        return x, f, Factor(Q + self.At @ sp.diags(w) @ self.A, self._perm)

    def log_marginal(self, theta, require_converged=False):
        x, f, lu = self.mode(theta)
        if require_converged and not self._converged:
            # an unconverged inner solve gives a meaningless (often spuriously high)
            # Laplace value; reject it and keep the previous warm start
            raise FloatingPointError("inner Newton iterations did not converge")
        self._x = x
        ld_post = lu.logdet()
        return f + 0.5 * self.logdet_Q(theta) - 0.5 * ld_post

    # ---- outer optimisation -------------------------------------------------
    def fit(self, theta0=None, verbose=False, maxiter=200):
        t0 = time.time()
        theta0 = (np.concatenate([b.theta0() for b in self.blocks])
                  if theta0 is None else np.asarray(theta0, float))
        if self.n_theta == 0:
            self.theta = theta0
        else:
            def nlp(th):
                self.n_evals += 1
                try:
                    v = -(self.log_marginal(th, require_converged=True) + self.log_prior(th))
                except Exception:
                    return 1e100
                if verbose:
                    print(np.round(th, 3), round(v, 3), flush=True)
                return v if np.isfinite(v) else 1e100

            bounds = [bd for b in self.blocks for bd in b.bounds()]
            res = minimize(nlp, theta0, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=maxiter, eps=1e-4))
            self.theta, self.opt = res.x, res
        self.x_hat, self.f_hat, self.lu = self.mode(self.theta)
        self.log_ml = self.log_marginal(self.theta)
        self.fit_seconds = time.time() - t0
        return self

    # ---- posterior summaries ------------------------------------------------
    def block_mean(self, k):
        return self.x_hat[self.slices[k]]

    def block_cov_dense(self, k):
        """Posterior covariance of a (small) block, e.g. the fixed effects."""
        s = self.slices[k]
        E = np.zeros((self.dim, s.stop - s.start))
        E[np.arange(s.start, s.stop), np.arange(s.stop - s.start)] = 1.0
        return self.lu.solve(E)[s]

    def sample(self, n, rng: np.random.Generator, batch=25):
        """Draws from N(x*, (Q + H)^{-1}) via the perturbation trick:
        x = x* + (Q + H)^{-1} (R' e1 + A' W^{1/2} e2), in batches of `batch` draws so
        that the observation-level noise never needs more than len(y) x batch memory."""
        R = self.R(self.theta)
        eta = self.offset + self.A @ self.x_hat
        _, w = self._grad_w(eta)
        sw = np.sqrt(w)[:, None]
        out = np.empty((self.dim, n))
        for a in range(0, n, batch):
            k = min(batch, n - a)
            e1 = rng.standard_normal((self.dim, k))
            e2 = rng.standard_normal((len(self.y), k)) * sw
            out[:, a:a + k] = self.x_hat[:, None] + self.lu.solve(R.T @ e1 + self.At @ e2)
        return out

    def block_sd(self, k, n=200, rng=None):
        rng = rng or np.random.default_rng(0)
        return self.sample(n, rng)[self.slices[k]].std(axis=1)

    def hyper(self):
        out = {}
        for b, t in zip(self.blocks, self._split(self.theta)):
            out.update(b.describe(t))
        return out

    def fixed_summary(self, k=0):
        b = self.blocks[k]
        m = self.block_mean(k)
        sd = np.sqrt(np.diag(self.block_cov_dense(k)))
        return {nm: (mm, ss) for nm, mm, ss in zip(b.names, m, sd)}

    def predict_eta(self, designs_new, offset_new=None, n_samples=0, rng=None):
        """Linear predictor for new rows. Returns posterior mean and, if
        n_samples > 0, posterior sd of eta (from latent samples)."""
        A_new = sp.hstack([sp.csr_matrix(D) for D in designs_new], format="csr")
        off = 0.0 if offset_new is None else offset_new
        mean = off + A_new @ self.x_hat
        if n_samples:
            S = self.sample(n_samples, rng or np.random.default_rng(1))
            sd = (A_new @ S).std(axis=1)
            return mean, sd
        return mean, None


def prob_from_eta(mean, sd=None):
    """E[expit(eta)] for eta ~ N(mean, sd^2) (probit-matching approximation)."""
    if sd is None:
        return expit(mean)
    return expit(mean / np.sqrt(1 + np.pi * sd ** 2 / 8))
