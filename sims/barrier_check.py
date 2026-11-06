"""Remark 1 check: boundary treatments on a 60 x 30 lattice of 0.5-degree cells with a
sea strait of 2, 4 or 6 columns (about 110, 220, 330 km) in the middle. Reports the marginal variance of a
coastal land cell and of an inland cell (nominal 1) and the correlation between the
two coastal cells facing each other across the strait, from 4,000 exact draws.

  land only   strait removed from the lattice (reflecting coastline)
  sea = land  strait cells kept as ordinary lattice cells without data
  barrier     strait cells kept with range barrier_frac * rho (barrier model)
"""
from common import *  # noqa: F401,F403
from sc.gmrf import SphereLattice

RANGE, SD, NDRAW = 400.0, 1.0, 4000


def main(width=6):
    L = SphereLattice(10.25 + 0.5 * np.arange(60), -4.75 + 0.5 * np.arange(30))
    c0, c1 = 30 - width // 2, 30 - width // 2 + width
    sea = (L.col >= c0) & (L.col < c1)
    th = np.log([RANGE, SD])
    rng = np.random.default_rng(0)
    coast, across, inland = (15, c0 - 1), (15, c1), (15, 10)
    rows = []

    mask = np.ones(L.mask.shape, bool); mask[:, c0:c1] = False
    L1 = SphereLattice(L.lon_c, L.lat_c, mask=mask)
    S = MaternBlock(L1).sample(th, rng, NDRAW); v = S.var(1); i = L1.index
    rows.append(dict(width_cells=width, treatment="land only", var_coast=v[i[coast]], var_inland=v[i[inland]],
                     corr_across=np.corrcoef(S[i[coast]], S[i[across]])[0, 1]))
    for name, kw in [("sea = land", {}), ("barrier 0.1", dict(barrier=sea, barrier_frac=0.1)),
                     ("barrier 0.2", dict(barrier=sea, barrier_frac=0.2)),
                     ("barrier 0.5", dict(barrier=sea, barrier_frac=0.5))]:
        S = MaternBlock(L, **kw).sample(th, rng, NDRAW); v = S.var(1); i = L.index
        rows.append(dict(width_cells=width, treatment=name, var_coast=v[i[coast]], var_inland=v[i[inland]],
                         corr_across=np.corrcoef(S[i[coast]], S[i[across]])[0, 1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = pd.concat([main(w) for w in (2, 4, 6)], ignore_index=True)
    res.to_csv(os.path.join(RESULTS, "barrier_check.csv"), index=False)
    print(res.round(3).to_string(index=False))
