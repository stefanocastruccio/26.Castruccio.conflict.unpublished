"""How many event cell-weeks have two or more events? (Discussion: Bernoulli vs count/Cox models.)

Counts per cell-week: conflict = Battles + Explosions/Remote violence + Violence against civilians;
unrest = Protests + Riots (the event types behind the binary outcomes of fit_glmm.prepare). Uses
the repaired one-row-per-cell-week table after coverage and biome exclusions ($CLEAN), restricted
to a region (ame: the Africa + Middle East box without its European countries; west_sahel).
Writes aggregate shares only (no row-level data) to a JSON.

Usage: python count_multiplicity.py $CLEAN --land $COUNTRIES --out results_real/multiplicity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
import pandas as pd

from fit_bivariate import REGIONS, region_rows

TYPES = {"conflict": ["Battles", "ExplosionsRemote_violence", "Violence_against_civilians"],
         "unrest": ["Protests", "Riots"]}
BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 10**9)]


def summarise(n):
    ev = n[n > 0]
    out = {"cell_weeks": int(len(n)), "event_cell_weeks": int(len(ev)),
           "share_event": float(len(ev) / len(n)) if len(n) else None,
           "events": int(ev.sum())}
    if len(ev):
        out["share_2plus_among_event"] = float((ev >= 2).mean())
        out["share_of_events_in_2plus_cellweeks"] = float(ev[ev >= 2].sum() / ev.sum())
        out["mean_events_per_event_cellweek"] = float(ev.mean())
        out["p99_events_per_event_cellweek"] = float(np.quantile(ev, 0.99))
        out["max"] = int(ev.max())
        out["distribution_among_event"] = {f"{a}" if a == b else (f"{a}+" if b > 10**8 else f"{a}-{b}"):
                                           float(((ev >= a) & (ev <= b)).mean()) for a, b in BINS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--land", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--regions", default="ame,west_sahel")
    a = ap.parse_args()
    cols = ["grid_id", "week_start", "lon", "lat", "ISO_A3"] + sum(TYPES.values(), [])
    raw = pd.read_parquet(a.table, columns=cols)
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    res = {}
    for reg in a.regions.split(","):
        d = region_rows(raw, reg, REGIONS[reg], a.land)
        r = {}
        for o, ty in TYPES.items():
            n = d[ty].fillna(0).sum(axis=1).to_numpy()
            r[o] = summarise(n)
            byc = (pd.DataFrame({"iso": d["ISO_A3"].astype(str).to_numpy(), "n": n})
                   .query("n > 0").groupby("iso")["n"]
                   .agg(event_cell_weeks="size", share_2plus=lambda x: float((x >= 2).mean())))
            r[o]["top_countries_by_event_cellweeks"] = (byc.sort_values("event_cell_weeks", ascending=False)
                                                        .head(10).reset_index().to_dict(orient="records"))
        # both outcomes in the same cell-week
        nc = d[TYPES["conflict"]].fillna(0).sum(axis=1).to_numpy()
        nu = d[TYPES["unrest"]].fillna(0).sum(axis=1).to_numpy()
        r["years"] = [str(d["week_start"].min().date()), str(d["week_start"].max().date())]
        r["share_both"] = float(((nc > 0) & (nu > 0)).mean())
        res[reg] = r
        print(reg, json.dumps({o: {k: r[o][k] for k in ("event_cell_weeks", "share_2plus_among_event",
                                                         "share_of_events_in_2plus_cellweeks", "distribution_among_event")}
                               for o in TYPES}, default=float), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
