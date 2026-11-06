"""Direct checks on the original partition files (STUDENT_CODE_REVIEW.md items 1 and 1b).

1. Which columns differ across the copies of a duplicated (grid_id, week_start)?
   For every column, the share of duplicated cell-weeks whose copies disagree.
2. Leakage: the 4-week event counts computed exactly as in the original
   feature_engineering (rows sorted by cell and week, then shift(1).rolling(4) row by
   row) are compared with the true counts over the previous four weeks (one row per
   cell-week). Reported: the share of rows whose original count differs from the true
   one, the share of event weeks whose original count includes the current week's
   events while the true past count is zero, and the AUC-PR of each count as a
   predictor of the outcome (a model-free view of the leakage).

Usage: python check_duplicates.py PARTS_DIR OUT.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EVENTS = {"Conflict": ["Battles", "ExplosionsRemote_violence", "Violence_against_civilians"],
          "Unrest": ["Riots", "Protests"]}
KEY = ["grid_id", "week_start"]


def main(parts, out):
    files = sorted(glob.glob(os.path.join(parts, "*.csv"))) or sorted(glob.glob(os.path.join(parts, "*.parquet")))
    read = (lambda f: pd.read_parquet(f)) if files[0].endswith(".parquet") else pd.read_csv
    differ, n_dup_pairs, keep = None, 0, []
    ev_cols = sum(EVENTS.values(), [])
    for f in files:
        df = read(f)
        g = df.groupby(KEY, sort=False)
        size = g["grid_id"].transform("size")
        dup = df[size > 1]
        if len(dup):
            nun = dup.groupby(KEY).nunique(dropna=False)
            d = (nun > 1).sum()
            differ = d if differ is None else differ.add(d, fill_value=0)
            n_dup_pairs += len(nun)
        keep.append(df[KEY + ev_cols])
        print(os.path.basename(f), flush=True)
    res = {"duplicated_cell_weeks": int(n_dup_pairs),
           "share_of_duplicated_cell_weeks_whose_copies_differ": {
               k: float(v / max(n_dup_pairs, 1)) for k, v in (differ if differ is not None else pd.Series(dtype=float)).items()
               if v > 0}}

    df = pd.concat(keep, ignore_index=True); del keep
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    first = df.drop_duplicates(KEY)
    for out_name, cols in EVENTS.items():
        tot = df[cols].sum(axis=1)
        # original: row-based shift/rolling within cell over the duplicated rows
        orig = tot.groupby(df["grid_id"]).shift(1).rolling(4).sum()
        # truth: one row per cell-week
        t1 = first[cols].sum(axis=1)
        true1 = t1.groupby(first["grid_id"]).shift(1).rolling(4).sum()
        true = df[KEY].merge(pd.DataFrame({"grid_id": first["grid_id"], "week_start": first["week_start"],
                                           "true": true1.to_numpy()}), on=KEY, how="left")["true"]
        y = (tot > 0).astype(int).to_numpy()
        ok = orig.notna().to_numpy() & true.notna().to_numpy()
        o, tr = orig.to_numpy()[ok], true.to_numpy()[ok]
        yy = y[ok]
        res[out_name] = {
            "share_rows_original_count_differs": float(np.mean(o != tr)),
            "share_event_weeks_with_leak": float(np.mean((o > 0) & (tr == 0) & (yy == 1)) / max(yy.mean(), 1e-12)),
            "auc_pr_original_count": float(average_precision_score(yy, o)),
            "auc_pr_true_count": float(average_precision_score(yy, tr)),
            "rows": int(ok.sum())}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
