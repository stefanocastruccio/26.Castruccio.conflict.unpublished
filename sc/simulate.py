"""Simulator that mimics the structure of the ACLED grid-cell x week data.

Truth (logit scale) for cell s in country c(s) and biome b(s), week t:

  eta_st = beta0 + beta_pop * pop_s
           + delta * log1p(sum_{l=1..4} y_{s,t-l})          (true state dependence)
           + [gamma_{b(s)} + eta_svc(s)] * temp_anom_st       (biome / spatially varying climate effect)
           + beta_prec * prec_anom_st
           + beta_nat * Z_c(s)                               (national covariate, e.g. EPI)
           + u_c(s)                                          (country effect)
           + w(s)                                            (persistent spatial field)
           + v_st                                            (AR(1) x Matern space-time field)

Z_c can be made correlated with the country mean of w + u ("spatial confounding",
`nat_conf`), mimicking national indicators that track unobserved state capacity.
Two extra national covariates with no effect play the role of the other
country-level indicators, which (as in the paper) let trees identify countries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from scipy.special import expit

from .gmrf import SphereLattice, MaternBlock, SpaceTimeAR1Block


@dataclass
class SimConfig:
    nx: int = 40
    ny: int = 30
    res: float = 0.5
    lon0: float = 10.25
    lat0: float = -4.75
    T_train: int = 104
    T_test: int = 52
    burn: int = 12
    n_countries: int = 14
    n_biomes: int = 4
    beta0: float = -4.0
    beta_pop: float = 0.5
    delta: float = 0.6
    gamma_temp: tuple = (-0.30, -0.10, 0.10, 0.30)   # per biome, cold -> tropical
    svc_sd: float = 0.0
    svc_range: float = 600.0
    beta_prec: float = -0.15
    beta_nat: float = 0.0
    nat_conf: float = 0.0
    country_sd: float = 0.3
    field_sd: float = 1.0
    field_range: float = 400.0
    st_sd: float = 0.0
    st_range: float = 300.0
    st_rho: float = 0.8
    dgp: str = "gp"          # "gp": exact continuous Matern; "lattice": the fitted GMRF itself
    seed: int = 0


@dataclass
class SimData:
    cfg: SimConfig
    lattice: SphereLattice
    df: pd.DataFrame            # long format, one row per cell-week (after burn-in)
    cells: pd.DataFrame         # one row per cell with truths
    countries: pd.DataFrame     # one row per country with truths
    st_field: np.ndarray | None = None   # (T, n) space-time truth
    extra: dict = field(default_factory=dict)


_CHOL = {}


def matern_gp(L, range_km, sd, rng, k=1):
    """k draws at the cell centres of L from the exact continuous Matern (nu = 1)
    covariance on the sphere (great-circle distances): sd^2 x K1(x), x = sqrt(8) d / range.
    Used to generate data without the 'inverse crime' of simulating from the lattice
    GMRF that is then fitted."""
    from scipy.special import kv
    from .diagnostics import haversine
    key = (id(L), L.n, range_km)
    if key not in _CHOL:
        D = haversine(L.lon[:, None], L.lat[:, None], L.lon[None], L.lat[None])
        x = np.sqrt(8.0) / range_km * D
        with np.errstate(invalid="ignore"):
            C = np.where(x > 0, x * kv(1, np.maximum(x, 1e-12)), 1.0)
        if len(_CHOL) > 8:
            _CHOL.clear()
        _CHOL[key] = np.linalg.cholesky(C + 1e-8 * np.eye(L.n))
    return sd * _CHOL[key] @ rng.standard_normal((L.n, k))


def _standardize(v):
    return (v - v.mean()) / v.std()


def simulate(cfg: SimConfig) -> SimData:
    rng = np.random.default_rng(cfg.seed)
    lon = cfg.lon0 + cfg.res * np.arange(cfg.nx)
    lat = cfg.lat0 + cfg.res * np.arange(cfg.ny)
    L = SphereLattice(lon, lat)
    n = L.n

    def matern(range_km, sd, k=1):
        if cfg.dgp == "lattice":
            return MaternBlock(L).sample(np.log([range_km, sd]), rng, k)
        return matern_gp(L, range_km, sd, rng, k)

    # --- geography --------------------------------------------------------
    seeds = rng.choice(n, cfg.n_countries, replace=False)
    d2 = (L.lon[:, None] - L.lon[seeds][None]) ** 2 + (L.lat[:, None] - L.lat[seeds][None]) ** 2
    country = np.argmin(d2, axis=1)
    # biomes: latitude bands with a wiggly boundary, ordered cold(0) -> tropical(B-1)
    score = L.lat + 2.0 * matern(800, 1.0).ravel()
    q = np.quantile(score, np.linspace(0, 1, cfg.n_biomes + 1)[1:-1])
    biome = (cfg.n_biomes - 1) - np.searchsorted(q, score)   # high lat -> 0 (cold)
    pop = _standardize(matern(250, 1.0).ravel() + 0.6 * rng.standard_normal(n))

    # --- latent effects -----------------------------------------------------
    w = matern(cfg.field_range, cfg.field_sd).ravel() if cfg.field_sd > 0 else np.zeros(n)
    u = cfg.country_sd * rng.standard_normal(cfg.n_countries)
    svc = matern(cfg.svc_range, cfg.svc_sd).ravel() if cfg.svc_sd > 0 else np.zeros(n)
    gamma = np.asarray(cfg.gamma_temp)[biome]

    # national covariates: Z0 (possibly confounded with country-level latent risk), Z1, Z2 nuisance
    latent_c = np.array([w[country == c].mean() if np.any(country == c) else 0.0
                         for c in range(cfg.n_countries)]) + u
    latent_c = _standardize(latent_c)
    Z = rng.standard_normal((cfg.n_countries, 3))
    Z[:, 0] = cfg.nat_conf * latent_c + np.sqrt(1 - cfg.nat_conf ** 2) * Z[:, 0]
    Z = (Z - Z.mean(0)) / Z.std(0)

    # --- weekly climate anomalies: AR(1) in time of smooth Matern fields ------
    Ttot = cfg.burn + cfg.T_train + cfg.T_test
    E_t = matern(800, 1.0, Ttot)
    E_p = matern(500, 1.0, Ttot)
    temp = np.zeros((Ttot, n)); prec = np.zeros((Ttot, n))
    a, b = 0.6, np.sqrt(1 - 0.36)
    temp[0], prec[0] = E_t[:, 0], E_p[:, 0]
    for t in range(1, Ttot):
        temp[t] = a * temp[t - 1] + b * E_t[:, t]
        prec[t] = 0.3 * prec[t - 1] + np.sqrt(1 - 0.09) * E_p[:, t]
    temp /= temp.std(); prec /= prec.std()

    # --- space-time field -----------------------------------------------------
    v = np.zeros((Ttot, n))
    if cfg.st_sd > 0:
        eps = matern(cfg.st_range, cfg.st_sd, Ttot).T
        v[0] = eps[0]
        s = np.sqrt(1 - cfg.st_rho ** 2)
        for t in range(1, Ttot):
            v[t] = cfg.st_rho * v[t - 1] + s * eps[t]

    # --- sequential generation (lags) -----------------------------------------
    static = (cfg.beta0 + cfg.beta_pop * pop + cfg.beta_nat * Z[country, 0]
              + u[country] + w)
    Y = np.zeros((Ttot, n), int)
    LAG = np.zeros((Ttot, n))
    P = np.zeros((Ttot, n))
    for t in range(Ttot):
        lag = Y[max(0, t - 4):t].sum(0) if t > 0 else np.zeros(n)
        LAG[t] = lag
        eta = (static + cfg.delta * np.log1p(lag) + (gamma + svc) * temp[t]
               + cfg.beta_prec * prec[t] + v[t])
        P[t] = expit(eta)
        Y[t] = rng.binomial(1, P[t])

    keep = slice(cfg.burn, Ttot)
    T = cfg.T_train + cfg.T_test
    week = np.repeat(np.arange(T), n)
    cell = np.tile(np.arange(n), T)
    df = pd.DataFrame({
        "cell": cell, "week": week,
        "lon": L.lon[cell], "lat": L.lat[cell],
        "country": country[cell], "biome": biome[cell],
        "abs_lat": np.abs(L.lat[cell]),
        "pop": pop[cell],
        "temp_anom": temp[keep].ravel(), "prec_anom": prec[keep].ravel(),
        "Z0": Z[country, 0][cell], "Z1": Z[country, 1][cell], "Z2": Z[country, 2][cell],
        "lag4": LAG[keep].ravel(), "y": Y[keep].ravel(), "p_true": P[keep].ravel(),
    })
    df["log_lag4"] = np.log1p(df["lag4"])
    # raw temperature, as a boosting model would see it: climatology (warmer towards the
    # "tropical" biomes) + seasonal cycle + the anomaly that carries the effect
    clim = 30.0 - 1.2 * (L.lat - L.lat.min())
    season = (1.0 + 0.15 * np.abs(L.lat))[None, :] * np.cos(2 * np.pi * np.arange(Ttot) / 52.0)[:, None]
    df["temp_raw"] = (clim[None, :] + season + 2.5 * temp)[keep].ravel()
    df["train"] = df["week"] < cfg.T_train
    cells = pd.DataFrame({"cell": np.arange(n), "lon": L.lon, "lat": L.lat,
                          "country": country, "biome": biome, "pop": pop,
                          "w": w, "svc": svc, "beta_temp": gamma + svc})
    countries = pd.DataFrame({"country": np.arange(cfg.n_countries), "u": u,
                              "Z0": Z[:, 0], "Z1": Z[:, 1], "Z2": Z[:, 2],
                              "latent": latent_c})
    return SimData(cfg, L, df, cells, countries, v[keep] if cfg.st_sd > 0 else None)
