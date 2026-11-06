"""S0 -- Accuracy of the Laplace engine and of the finite-volume SPDE field.

(a) Marginal variance of the lattice Matern field (interior vs boundary cells).
(b) Bernoulli model  logit p = b0 + b1 x + w(s) on 1200 cells, with w drawn from the
    exact continuous Matern process (not from the fitted lattice GMRF). Two settings
    (CONFIG): "common" (b0 = -2, 40 weeks, prevalence about 15%) and "rare" (b0 = -4.5,
    104 weeks, prevalence about 2%, as in the ACLED data). Recovery of (range, sd) and
    b1, and frequentist coverage of 95% intervals for b1 and for b0 + w(s).
"""
from common import *  # noqa: F401,F403

from scipy.special import expit
from sc.gmrf import SphereLattice
from sc.simulate import matern_gp

N_REPS = int(os.environ.get("N_REPS", 50))
CONFIGS = {"common": (-2.0, 40), "rare": (-4.5, 104)}
RANGE, SD, B1 = 400.0, 1.0, 0.5


def lattice():
    return SphereLattice(10.25 + 0.5 * np.arange(40), -4.75 + 0.5 * np.arange(30))


def one(arg):
    config, seed = arg
    B0, T = CONFIGS[config]
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    w = matern_gp(L, RANGE, SD, rng).ravel()
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
    return dict(config=config, seed=seed, prevalence=float(y.mean()), range_hat=h["field.range_km"], sd_hat=h["field.sd"],
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
    args = [(c, s) for c in CONFIGS for s in range(1, N_REPS + 1)]
    res = pd.DataFrame(run_parallel(one, args, checkpoint=os.path.join(RESULTS, "s0_ckpt.jsonl")))
    if os.environ.get("SHARD"):
        sys.exit(0)
    res.to_csv(os.path.join(RESULTS, "s0_raw.csv"), index=False)
    summ = res.drop(columns="seed").groupby("config").agg(["mean", "std"]).T
    summ.to_csv(os.path.join(RESULTS, "s0_summary.csv"))
    pd.Series(var).to_csv(os.path.join(RESULTS, "s0_variance.csv"))
    print(var); print(summ.round(3).to_string())
