"""Task 40 checks (aggregate only): results_real/table_global_checks.json.

(a) rows and event cell-weeks by region and year (from core_report.json);
(b) 2018-2024, cell-weeks present in $CLEAN: agreement of Conflict and Unrest with $CLEAN (share equal; disagreements by country),
    and of the 5 per-type counts with Luca's partitions (first copy, $WORK/multiplicity_input.parquet);
(c) share of events with geo_precision 1 by region (from core_report.json);
(d) disk use of $WORK/table_global.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "covariates"))
import common as C  # noqa: E402

T = os.path.join(C.WORK, "table_global")
TYPES = ["Battles", "ExplosionsRemote_violence", "Violence_against_civilians", "Protests", "Riots"]


def main():
    rep = json.load(open(os.path.join(T, "core_report.json")))
    out = {"a_rows_events_by_region_year": rep["by_region_year"], "rows_total": rep["rows_total"],
           "c_geo_precision1_share_by_region": rep["geo_precision1_share_by_cell_region"],
           "events": {k: rep[k] for k in ("events_raw", "events_used", "events_dropped_no_cell", "events_moved_to_neighbour_cell",
                                          "events_outside_week_range", "events_unknown_type")},
           "dropped_share_by_acled_region": rep["dropped_share_by_acled_region"], "weeks": rep["weeks"],
           "coverage_source_counts": rep["coverage_source_counts"]}
    core = pd.concat([pd.read_parquet(os.path.join(T, "core", f"year={y}", "part.parquet"),
                                      columns=["grid_id", "week_start", "ISO_A3", "Conflict", "Unrest"] + TYPES) for y in range(2018, 2025)])
    cl = pd.read_parquet(C.CLEAN, columns=["grid_id", "week_start", "ISO_A3", "Conflict", "Unrest"])
    cl = cl[(cl.week_start.dt.year >= 2018) & (cl.week_start.dt.year <= 2024)]
    m = cl.merge(core, on=["grid_id", "week_start"], how="left", suffixes=("_clean", ""))
    b = {"clean_cell_weeks_2018_2024": int(len(cl)), "missing_in_global_table": int(m.Conflict.isna().sum()),
         "missing_by_country_top": m[m.Conflict.isna()].ISO_A3_clean.value_counts().head(15).to_dict()}
    mm = m.dropna(subset=["Conflict"])
    for v in ("Conflict", "Unrest"):
        eq = mm[v + "_clean"].astype(int) == mm[v].astype(int)
        b[f"{v}_share_equal"] = float(eq.mean())
        b[f"{v}_clean_rate"] = float(mm[v + "_clean"].mean()); b[f"{v}_global_rate"] = float(mm[v].mean())
        b[f"{v}_only_in_clean"] = int(((mm[v + "_clean"] == 1) & (mm[v] == 0)).sum())
        b[f"{v}_only_in_global"] = int(((mm[v + "_clean"] == 0) & (mm[v] == 1)).sum())
        dis = mm[~eq].groupby("ISO_A3_clean").size().sort_values(ascending=False)
        b[f"{v}_disagreements_by_country_top"] = dis.head(15).to_dict()
    mp = pd.read_parquet(os.path.join(C.WORK, "multiplicity_input.parquet"), columns=["grid_id", "week_start"] + TYPES)
    mp = mp[(mp.week_start.dt.year >= 2018) & (mp.week_start.dt.year <= 2024)]
    j = mp.merge(core[["grid_id", "week_start"] + TYPES], on=["grid_id", "week_start"], how="left", suffixes=("_luca", ""))
    j = j.dropna(subset=[TYPES[0]])
    b["partition_cell_weeks_matched"] = int(len(j))
    for t in TYPES:
        a_, g_ = j[t + "_luca"].fillna(0).to_numpy(float), j[t].to_numpy(float)
        b[f"{t}_count_share_equal"] = float(np.mean(a_ == g_))
        b[f"{t}_count_pearson"] = float(np.corrcoef(a_, g_)[0, 1]) if a_.std() > 0 and g_.std() > 0 else None
        b[f"{t}_total_luca_vs_global"] = [float(a_.sum()), float(g_.sum())]
    out["b_agreement_2018_2024"] = b
    out["d_disk"] = subprocess.run(["du", "-sh"] + [os.path.join(T, x) for x in os.listdir(T)], capture_output=True, text=True).stdout
    od = os.path.join(os.path.dirname(__file__), "..", "..", "results_real"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "table_global_checks.json"), "w"), indent=1, default=str)
    print(json.dumps(b, indent=1, default=str))


if __name__ == "__main__":
    main()
