"""Restrict every covariate output under $WORK/covariates/grid to the cells of the five study regions (cell_region.parquet),
and the national table to their countries. Idempotent. Task 35 scope (2026-11-09)."""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

keep = set(C.region_cells())
cr = pd.read_parquet(os.path.join(C.GRID, "cell_region.parquet"))
before = sum(os.path.getsize(f) for f in glob.glob(os.path.join(C.GRID, "**", "*.parquet"), recursive=True))
for f in sorted(glob.glob(os.path.join(C.GRID, "**", "*.parquet"), recursive=True)):
    if f.endswith("cell_region.parquet"):
        continue
    d = pd.read_parquet(f)
    if "grid_id" in d:
        d2 = d[d.grid_id.isin(keep)]
    elif "iso3" in d:   # national: countries of the in-region cells (CLEAN ISO_A3 codes)
        d2 = d[d.iso3.isin(set(cr.loc[cr.region.notna(), "ISO_A3"]))]
    else:
        continue
    if len(d2) < len(d):
        d2.to_parquet(f + ".tmp", index=False); os.replace(f + ".tmp", f)
after = sum(os.path.getsize(f) for f in glob.glob(os.path.join(C.GRID, "**", "*.parquet"), recursive=True))
print(f"grid outputs: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB; {len(keep)} cells kept")
