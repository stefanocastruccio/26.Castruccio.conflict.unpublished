"""S0 -- Accuracy of the Laplace engine and of the finite-volume SPDE field.

(a) Marginal variance of the lattice Matern field (interior vs boundary cells).
(b) Bernoulli model  logit p = b0 + b1 x + w(s), 40 weeks x 1200 cells:
    recovery of (range, sd), of b1, and frequentist coverage of 95% intervals
    for b1 and for b0 + w(s).
"""
from common import *  # noqa: F401,F403

from scipy.special import expit
from sc.gmrf import SphereLattice

N_REPS = int(os.environ.get("N_REPS", 50))
RANGE, SD, B0, B1, T = 400.0, 1.0, -2.0, 0.5, 40


def lattice():
    return SphereLattice(10.25 + 0.5 * np.arange(40), -4.75 + 0.5 * np.arange(30))


def one(seed):
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    mb = MaternBlock(L)
    w = mb.sample(np.log([RANGE, SD]), rng).ravel()
    cell = np.tile(np.arange(n), T)
    x = rng.standard_normal(n * T)
    y = rng.binomial(1, expit(B0 + B1 * x + w[cell]))
    m = LatentGaussianModel(y, [FixedBlock(2, ["b0", "b1"]), MaternBlock(L)],
                            [np.c_[np.ones(n * T), x], de.index_design(cell, n)]).fit()
    est, se = m.fixed_summary()["b1"]
    S = m.sample(500, np.random.default_rng(seed + 10_000))
    lin = S[0] + S[2:]
    lo, hi = np.quantile(lin, [0.025, 0.975], axis=1)
    h = m.hyper()
    return dict(seed=seed, range_hat=h["field.range_km"], sd_hat=h["field.sd"],
                b1_hat=est, b1_cover=float(abs(est - B1) < 1.96 * se),
                field_cover=float(np.mean((lo <= B0 + w) & (B0 + w <= hi))),
                field_corr=float(np.corrcoef(m.block_mean(1), w)[0, 1]),
                seconds=m.fit_seconds)


if __name__ == "__main__":
    L = lattice()
    S = MaternBlock(L).sample(np.log([RANGE, SD]), np.random.default_rng(0), 2000)
    v = S.var(axis=1).reshape(L.mask.shape)
    var = {"interior (>= 8 cells from edge)": float(np.median(v[8:-8, 8:-8])),
           "edge cells": float(np.median(np.r_[v[0], v[-1], v[:, 0], v[:, -1]]))}
    res = pd.DataFrame(run_parallel(one, range(1, N_REPS + 1),
                                    checkpoint=os.path.join(RESULTS, "s0_ckpt.jsonl")))
    res.to_csv(os.path.join(RESULTS, "s0_raw.csv"), index=False)
    summ = res.drop(columns="seed").agg(["mean", "std"]).T
    summ.to_csv(os.path.join(RESULTS, "s0_summary.csv"))
    pd.Series(var).to_csv(os.path.join(RESULTS, "s0_variance.csv"))
    print(var); print(summ.round(3).to_string())
