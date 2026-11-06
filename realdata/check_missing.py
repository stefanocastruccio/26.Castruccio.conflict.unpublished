"""Missingness of national indicators and other covariates in the repaired table.

Reports, for the whole table and for the Africa + Middle East box (lon -20..65,
lat -36..45): the share of rows with each covariate missing, and the countries
with the most missing cell-weeks for each national indicator.

Usage: python check_missing.py CLEAN.parquet OUT.json
"""
import json
import sys

import pandas as pd

NAT = ["EPI_index", "v2x_libdem", "GHG_pc_t", "MF_pc_t", "ECO_pc_gha", "gdp", "gdp_pc", "electricity_access", "phase_3"]


def main(path, out):
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("grid_id", "lon", "lat")]
    box = df.lon.between(-20, 65) & df.lat.between(-36, 45)
    res = {}
    for name, d in [("all", df), ("africa_me", df[box])]:
        miss = d[cols].isna().mean()
        res[name] = {"rows": int(len(d)), "share_missing": {c: float(v) for c, v in miss.items() if v > 0}}
        res[name]["top_countries_missing"] = {
            c: {k: int(v) for k, v in d[c].isna().groupby(d["ISO_A3"]).sum().nlargest(12).items() if v > 0}
            for c in NAT if c in d}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v["share_missing"] for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
