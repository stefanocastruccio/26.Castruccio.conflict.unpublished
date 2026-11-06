"""Helpers that turn a long cell x week table into latent-model design matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def fixed_design(df: pd.DataFrame, cols: list[str], intercept=True, factors=()):
    """Dense fixed-effect design. `factors` are (column, value-col or None) pairs:
    ('biome', None) adds treatment dummies for biome; ('biome', 'temp_anom')
    adds biome-specific slopes of temp_anom (one column per level, no main effect)."""
    blocks, names = [], []
    if intercept:
        blocks.append(np.ones((len(df), 1))); names.append("intercept")
    for c in cols:
        blocks.append(df[c].to_numpy(float)[:, None]); names.append(c)
    for fac, val in factors:
        levels = np.sort(df[fac].unique())
        if val is None:
            for lv in levels[1:]:
                blocks.append((df[fac].to_numpy() == lv).astype(float)[:, None])
                names.append(f"{fac}[{lv}]")
        else:
            x = df[val].to_numpy(float)
            for lv in levels:
                blocks.append(((df[fac].to_numpy() == lv) * x)[:, None])
                names.append(f"{val}:{fac}[{lv}]")
    return np.hstack(blocks), names


def index_design(idx, size, values=None):
    """Sparse n x size matrix with A[i, idx[i]] = values[i] (default 1)."""
    idx = np.asarray(idx, int)
    v = np.ones(len(idx)) if values is None else np.asarray(values, float)
    return sp.csr_matrix((v, (np.arange(len(idx)), idx)), shape=(len(idx), size))


def spacetime_design(cell, week, n_cells, T):
    return index_design(np.asarray(week) * n_cells + np.asarray(cell), n_cells * T)
