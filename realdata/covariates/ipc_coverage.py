"""Task 51 check: share of cell-weeks with IPC (phase3p_share not NaN) in the global table, 2018-2024, for the ame and west_sahel
boxes (cells selected as in fit_bivariate.region_rows, land file $COUNTRIES).
  python realdata/covariates/ipc_coverage.py [regions ...]   (default: ame west_sahel)"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common as C  # noqa: E402
from fit_bivariate import REGIONS, region_rows  # noqa: E402

regs = sys.argv[1:] or ["ame", "west_sahel"]
T = os.path.join(C.WORK, "table_global", "clean_global")
cl = C.cells()
for reg in regs:
    cells = set(region_rows(cl, reg, REGIONS[reg], os.environ["COUNTRIES"]).grid_id.astype(int))
    tot = have = 0; per = []
    for y in range(2018, 2025):
        d = pd.read_parquet(os.path.join(T, f"year={y}.parquet"), columns=["grid_id", "phase3p_share"])
        d = d[d.grid_id.isin(cells)]
        tot += len(d); have += int(d.phase3p_share.notna().sum()); per.append(f"{y} {d.phase3p_share.notna().mean():.3f}")
    print(f"{reg} ({len(cells)} cells): cell-weeks with IPC 2018-2024 {have / tot:.3f} ({have:,} of {tot:,}); by year: {', '.join(per)}", flush=True)
