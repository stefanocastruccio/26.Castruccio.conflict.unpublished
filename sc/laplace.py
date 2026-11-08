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


try:                                      # supernodal Cholesky (SuiteSparse), if installed
    from sksparse import cholmod as _cholmod
except ImportError:                       # pragma: no cover
    _cholmod = None

import os as _os
BACKEND = ("cholmod" if _cholmod is not None else "superlu") \
    if _os.environ.get("LAPLACE_BACKEND", "auto") == "auto" else _os.environ["LAPLACE_BACKEND"]


class Factor:
    """Factorisation of an SPD matrix Q + H.

    CHOLMOD backend (default when scikit-sparse is installed): supernodal Cholesky with
    the symbolic analysis (fill-reducing ordering) computed once per model and reused,
    since the sparsity pattern of Q + H does not depend on theta. SuperLU fallback:
    LU in symmetric mode with an optional fixed nested-dissection permutation."""

    def __init__(self, M, perm=None, symbolic=None):
        self.perm, self.kind = perm, "superlu"
        if symbolic is not None:
            self.kind = "cholmod"
            if hasattr(symbolic, "factorize"):          # scikit-sparse >= 0.5
                self.lu = symbolic.copy()
                self.lu.factorize(sp.csc_matrix(M))
            else:                                       # scikit-sparse <= 0.4
                self.lu = symbolic.cholesky(sp.csc_matrix(M))
        elif perm is None:
            self.lu = _lu(M)
        else:
            Mp = sp.csr_matrix(M)[perm][:, perm]
            self.lu = splu(sp.csc_matrix(Mp), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                           options=dict(SymmetricMode=True))

    def solve(self, b):
        if self.kind == "cholmod":
            return self.lu.solve(b) if hasattr(self.lu, "factorize") else self.lu.solve_A(b)
        if self.perm is None:
            return self.lu.solve(b)
        x = np.empty_like(b, dtype=float)
        x[self.perm] = self.lu.solve(np.ascontiguousarray(b[self.perm]))
        return x

    def logdet(self):
        if self.kind == "cholmod":
            return float(self.lu.logdet())
        return float(np.sum(np.log(np.abs(self.lu.U.diagonal()))))


_PAR_MODEL = None   # model shared with forked gradient workers (copy-on-write)


def _par_nlp(args):
    th, x0 = args
    m = _PAR_MODEL
    try:
        x, f, lu = m.mode(th, x0=x0)
        if not m._converged:
            return 1e100
        v = -(f + 0.5 * m.logdet_Q(th) - 0.5 * lu.logdet() + m.log_prior(th))
        return v if np.isfinite(v) else 1e100
    except Exception:
        return 1e100


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
        # Leading dense design columns (e.g. the fixed-effect covariates) are kept as a
        # dense array: H = A' W A is then assembled blockwise, with BLAS for the dense
        # block and its coupling to the latent effects and a sparse product only for the
        # truly sparse remainder (about 10x faster than a sparse product with dense rows).
        p_dense = 0
        for b, D in zip(blocks, self.designs):
            if D.shape[1] and D.nnz > 0.3 * D.shape[0] * D.shape[1] and D.shape[1] <= 2000:
                p_dense += D.shape[1]
            else:
                break
        self._pd = p_dense if 0 < p_dense < self.A.shape[1] else 0
        if self._pd:
            self._Xd = np.asarray(self.A[:, :self._pd].todense())
            self._S = self.A[:, self._pd:].tocsr()
            self._St = self._S.T.tocsr()
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
        self._symb = None      # CHOLMOD symbolic analysis, computed once (same reason)

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

    def H(self, w):
        """A' diag(w) A, with a fixed sparsity pattern (dense blocks are stored in full)."""
        if not self._pd:
            return self.At @ sp.diags(w) @ self.A
        p = self._Xd.shape[1]; m = self._S.shape[1]
        Hdd = np.zeros((p, p)); Hsd = np.zeros((m, p))
        step = 250_000                                       # row chunks: bounded memory per process
        for a in range(0, len(w), step):
            Xc = self._Xd[a:a + step]
            Xwc = Xc * w[a:a + step, None]
            Hdd += Xwc.T @ Xc
            Hsd += np.asarray(self._S[a:a + step].T @ Xwc)
        Hss = self._St @ sp.diags(w) @ self._S
        p, m = Hdd.shape[0], Hsd.shape[0]

        def full(Mat):                                       # keep explicit zeros: fixed pattern
            r, c = np.indices(Mat.shape)
            return sp.csr_matrix((Mat.ravel(), (r.ravel(), c.ravel())), shape=Mat.shape)
        return sp.bmat([[full(Hdd), full(Hsd.T)], [full(Hsd), Hss]], format="csc")

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
            M0 = Q + self.H(w)
            if BACKEND == "cholmod":
                self._perm = None
                if hasattr(_cholmod, "CholeskyFactor"):   # scikit-sparse >= 0.5
                    self._symb = _cholmod.CholeskyFactor(sp.csc_matrix(M0), supernodal_mode="supernodal")
                else:                                     # scikit-sparse <= 0.4
                    self._symb = _cholmod.analyze(sp.csc_matrix(M0), mode="supernodal")
            else:
                self._perm = nested_dissection(M0) if self.dim > 2000 else None
        self._converged = False
        for it in range(maxit):
            g_lik, w = self._grad_w(eta)
            grad = self.At @ g_lik - Q @ x
            lu = Factor(Q + self.H(w), self._perm, self._symb)
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
        return x, f, Factor(Q + self.H(w), self._perm, self._symb)

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
    def fit(self, theta0=None, verbose=False, maxiter=200, n_jobs=None, eps=1e-4):
        """Empirical Bayes: maximise log p(y | theta) + log p(theta) by L-BFGS-B with
        forward finite-difference gradients. With n_jobs > 1 (default: the environment
        variable FIT_JOBS, else 1) the n_theta perturbed evaluations of each gradient run
        in parallel forked processes, warm-started at the current mode."""
        import os
        t0 = time.time()
        # Outer tolerances (env overrides for the large real-data fits; defaults = scipy's).
        maxiter = int(os.environ.get("LAPLACE_MAXITER", maxiter))
        lopts = dict(maxiter=maxiter)
        if os.environ.get("LAPLACE_FTOL"):
            lopts["ftol"] = float(os.environ["LAPLACE_FTOL"])
        if os.environ.get("LAPLACE_GTOL"):
            lopts["gtol"] = float(os.environ["LAPLACE_GTOL"])
        theta0 = (np.concatenate([b.theta0() for b in self.blocks])
                  if theta0 is None else np.asarray(theta0, float))
        n_jobs = int(n_jobs or os.environ.get("FIT_JOBS", 1))
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
            if n_jobs > 1 and self.n_theta > 1:
                import multiprocessing as mp
                global _PAR_MODEL
                self.mode(theta0)                    # builds the ordering / symbolic analysis once
                _PAR_MODEL = self
                pool = mp.get_context("fork").Pool(min(n_jobs, self.n_theta))
                lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])

                def fg(th):
                    f0 = nlp(th)
                    x0 = self._x.copy()
                    # step inward at the upper bound
                    steps = [eps if th[i] + eps <= hi[i] else -eps for i in range(self.n_theta)]
                    pts = []
                    for i in range(self.n_theta):
                        t = th.copy(); t[i] += steps[i]; pts.append((t, x0))
                    fs = np.array(pool.map(_par_nlp, pts))
                    self.n_evals += self.n_theta
                    g = (fs - f0) / np.array(steps)
                    g[~np.isfinite(g) | (np.abs(fs) >= 1e99)] = 0.0
                    return f0, g

                try:
                    res = minimize(fg, theta0, method="L-BFGS-B", jac=True, bounds=bounds,
                                   options=lopts)
                finally:
                    pool.close(); pool.join(); _PAR_MODEL = None
            else:
                res = minimize(nlp, theta0, method="L-BFGS-B", bounds=bounds,
                               options=dict(lopts, eps=eps))
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
