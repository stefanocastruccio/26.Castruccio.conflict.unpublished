"""Task 45: conflict split by actor, per cell-week, for the three-outcome model (unrest, repression, armed violence).

  python realdata/table/actors.py     # batch job (crc/jobs/45_actors.sh); writes $WORK/table_global/actors/year=<y>.parquet

Events and cells exactly as in core.py (load_events). Outcomes, from ACLED's event type and the type of the first actor:
  Repression  Violence against civilians whose first actor is a state force (ACLED inter1 = 1 / "State forces").
  Armed       Battles + Explosions/Remote violence + Violence against civilians by any other first actor (rebels, militias,
              external forces, ...): organised armed violence.
  so Repression + Armed = core.py's n_conflict, event by event.
  n_excess_force  Protests with sub-event "Excessive force against protesters": counted separately; they stay in Unrest (as in
              core.py) and are NOT in Repression, so unrest -> repression is not partly the same event.
Columns: grid_id, week_start, n_repr, n_armed, n_excess_force, Repression = n_repr > 0, Armed = n_armed > 0, and the histories
Repression_rollsum_4, Armed_rollsum_4 (event counts over weeks t-4..t-1, like core.py's). Sparse: only cell-weeks with a non-zero
count or history; a missing row means zero. The fit sets the histories to NaN where core.py's are (the first 4 covered weeks).
Report: $WORK/table_global/actors_report.json (counts by actor type and region; check against core.py's n_conflict).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import core  # noqa: E402

STATE = {"1", "1.0", "state forces"}


def is_state(v):
    return str(v).strip().lower() in STATE


def main():
    t0 = time.time(); rep = {}
    cells = core.C.cells()[["grid_id", "lon", "lat", "ISO_A3", "row", "col"]].copy()
    lut = np.full(360 * 720, -1, np.int64); lut[cells.row * 720 + cells.col] = np.arange(len(cells))
    W = core.week_grid()
    ev = core.load_events(cells, lut, W, rep, extra_cols=("inter1", "sub_event_type"))
    st = ev.inter1.map(is_state)
    vac = ev.type == "Violence_against_civilians"
    ev["repr"] = vac & st
    ev["armed"] = ev.type.isin(["Battles", "ExplosionsRemote_violence"]) | (vac & ~st)
    ev["excess"] = (ev.type == "Protests") & (ev.sub_event_type.str.lower() == "excessive force against protesters")
    rep["inter1_values_violence_against_civilians"] = ev.loc[vac, "inter1"].astype(str).value_counts().head(20).to_dict()
    rep["events"] = {k: int(ev[k].sum()) for k in ("repr", "armed", "excess")}
    rep["events_conflict_core_definition"] = int(ev.type.isin(core.CONF).sum())
    rep["check_repr_plus_armed_equals_conflict"] = bool(rep["events"]["repr"] + rep["events"]["armed"]
                                                        == rep["events_conflict_core_definition"])
    rep["repr_share_of_vac_by_region"] = ev[vac].groupby("region")["repr"].mean().round(3).to_dict()
    g = ev.groupby(["cell", "widx"])
    piv = pd.DataFrame({"n_repr": g["repr"].sum(), "n_armed": g["armed"].sum(), "n_excess_force": g["excess"].sum()}).astype("int32")
    piv = piv[(piv > 0).any(axis=1)].reset_index()
    hist = pd.concat([piv[["cell", "widx", "n_repr", "n_armed"]].assign(widx=piv.widx + s) for s in range(1, 5)])
    hist = (hist.groupby(["cell", "widx"])[["n_repr", "n_armed"]].sum()
            .rename(columns={"n_repr": "Repression_rollsum_4", "n_armed": "Armed_rollsum_4"}).reset_index())
    df = piv.merge(hist, on=["cell", "widx"], how="outer")
    df = df[(df.widx >= 0) & (df.widx < len(W))]
    cnt = ["n_repr", "n_armed", "n_excess_force"]
    df[cnt] = df[cnt].fillna(0).astype("int32")
    hc = ["Repression_rollsum_4", "Armed_rollsum_4"]
    df[hc] = df[hc].fillna(0).astype("float32")
    df["Repression"] = (df.n_repr > 0).astype("int8"); df["Armed"] = (df.n_armed > 0).astype("int8")
    df.insert(0, "grid_id", cells.grid_id.to_numpy()[df.cell].astype("int32")); df.insert(1, "week_start", W[df.widx])
    df = df.drop(columns=["cell", "widx"])
    od = os.path.join(core.OUT, "actors"); os.makedirs(od, exist_ok=True)
    for y, d in df.groupby(df.week_start.dt.year):
        d.sort_values(["week_start", "grid_id"]).to_parquet(os.path.join(od, f"year={y}.parquet"), index=False)
    rep["rows"] = int(len(df)); rep["seconds"] = time.time() - t0
    json.dump(rep, open(os.path.join(core.OUT, "actors_report.json"), "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
