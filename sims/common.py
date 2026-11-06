"""Shared pieces for the simulation studies."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sc import design as de
from sc.gmrf import FixedBlock, IIDBlock, MaternBlock, SpaceTimeAR1Block
from sc.laplace import LatentGaussianModel

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# Features mimicking the paper's predictor set: no coordinates, but distance from
# the equator, biome, climate, population, recent history, national indicators.
XGB_FEATURES = ["pop", "abs_lat", "biome", "temp_anom", "prec_anom", "Z0", "Z1", "Z2", "lag4"]


def glmm(d: pd.DataFrame, sim, fixed_cols, field=False, country=False, spacetime=False,
         svc_col=None, factors=(), offset=None, T=None):
    """Build and fit a latent Gaussian model on rows `d` of a simulated dataset."""
    n = sim.lattice.n
    X, names = de.fixed_design(d, fixed_cols, factors=factors)
    blocks, designs = [FixedBlock(X.shape[1], names)], [X]
    if field:
        blocks.append(MaternBlock(sim.lattice, name="field"))
        designs.append(de.index_design(d["cell"], n))
    if country:
        nc = sim.cfg.n_countries
        blocks.append(IIDBlock(nc, name="country"))
        designs.append(de.index_design(d["country"], nc))
    if svc_col is not None:
        blocks.append(MaternBlock(sim.lattice, name="svc", sd_init=0.1,
                                  range_init=600.0, pc_sd=(0.5, 0.01)))
        designs.append(de.index_design(d["cell"], n, d[svc_col].to_numpy(float)))
    if spacetime:
        blocks.append(SpaceTimeAR1Block(sim.lattice, T, name="st"))
        designs.append(de.spacetime_design(d["cell"], d["week"], n, T))
    m = LatentGaussianModel(d["y"].to_numpy(), blocks, designs, offset=offset)
    return m.fit()


def block_index(m, name):
    return [b.name for b in m.blocks].index(name)


def run_parallel(fn, args, n_jobs=4, checkpoint=None):
    """Map fn over args in parallel. With `checkpoint` (a .jsonl path) every result
    is appended as soon as it finishes and already-finished args are skipped, so
    an interrupted study resumes where it stopped. Results come back in the order
    of `args`."""
    import json
    from joblib import Parallel, delayed
    args = list(args)
    if checkpoint is None:
        return Parallel(n_jobs=n_jobs)(delayed(fn)(a) for a in args)
    key = lambda a: json.dumps(a)
    done = {}
    if os.path.exists(checkpoint):
        with open(checkpoint) as fh:
            for line in fh:
                rec = json.loads(line)
                done[rec["key"]] = rec["result"]
    todo = [a for a in args if key(a) not in done]

    def wrapped(a):
        return key(a), fn(a)

    gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(delayed(wrapped)(a) for a in todo)
    with open(checkpoint, "a") as fh:
        for k, r in gen:
            fh.write(json.dumps({"key": k, "result": r}, default=float) + "\n"); fh.flush()
            done[k] = r
    return [done[key(a)] for a in args]
