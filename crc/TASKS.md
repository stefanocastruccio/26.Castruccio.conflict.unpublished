# Tasks for the CRC session

This file is how the cloud session ("Student paper spatial residual modeling") and the
CRC session talk to each other. The cloud session writes tasks here. The CRC session
runs them, writes results to `results_real/` and progress to `crc/STATUS.md`, and pushes.
Stefano has approved reading `/groups/scastruc/` and submitting jobs to the CRC queues.

Rules:

- **Never commit raw ACLED data, partition files, the Parquet tables or prediction
  files.** Only small summaries: the JSON/CSV/PNG files listed below. Large outputs stay in
  `$WORK` on the CRC.
- Run everything heavy through `qsub` (4 cores per job), not on the front-end.
- Before each push run `git pull --rebase`, and append to `crc/STATUS.md`: date/time,
  what ran, job ids, exit status, and the key numbers.
- If something fails, push the error (last ~40 lines of the job log) to `crc/STATUS.md`
  and stop that branch of work. The cloud session will fix the code and push.
- Adapt the SGE headers in `crc/jobs/` to the CRC's conventions as needed (queue names,
  `-pe smp 4`, GPU request, conda activation, Kerberos). Commit what you changed.

Paths (also set in `crc/jobs/env.sh`):

```bash
DATA=/groups/scastruc/lmenical/drew/data            # Luca's data (read-only)
WORK=/groups/scastruc/conflict_spatial_work         # large intermediate files
REPO=<your clone of stefanocastruccio/26.Castruccio.conflict.unpublished>
```

---

## Task 1: environment (front-end, light)

1. Clone this repo if you haven't already, and `git pull`.
2. In the conda env `/users/scastruc/envs/geo` (or a clone of it), install the
   requirements:
   ```bash
   pip install xgboost pyarrow pymetis geopandas pyogrio joblib matplotlib
   ```
   On a GPU node, check `python -c "import xgboost; print(xgboost.build_info()['USE_CUDA'])"`.
3. Check that these inputs exist:
   - `$DATA/world_map/ne_10m_admin_0_countries.shp`
   - `$DATA/socioeconomic/GDP.csv`
4. Report in STATUS: versions, the CUDA flag, and whether the two files exist.

## Task 2: clean tables (`crc/jobs/02_load.sh`, largemem queue)

Builds two tables from the 50 partitions:

- `$WORK/clean.parquet`: duplicates repaired (see `realdata/load_partitions.py`).
- `$WORK/orig.parquet`: duplicates kept, so the original analysis can be reproduced.

Report in STATUS the lines the loader prints: input rows, distinct cell-weeks, duplicated
share, cells, weeks. Also report the peak memory (`qacct -j <id> | grep maxvmem`).

## Task 3: XGBoost, original vs corrected (`crc/jobs/03_xgb.sh`, gpu queue)

This runs 4 jobs (mode × outcome) of `realdata/reproduce_xgb.py`. Each writes
`$WORK/xgb/metrics_<mode>_<outcome>.json` and `pred_<mode>_<outcome>.parquet`.

- Copy the four **metrics JSON files** to `results_real/xgb/` and push them.
- Put the main numbers in STATUS: `auc_pr`, `auc_pr_border_cells`,
  `auc_pr_interior_cells`, `persistence_auc_pr` and `rounds`.

## Task 4: spatial model, Africa + Middle East (`crc/jobs/04_spatial.sh`, largemem queue)

This runs `realdata/fit_acled.py` on `$WORK/clean.parquet` with `--region africa_me` for
each outcome. Copy each output folder's `summary.json`, `variogram.csv` and `field.csv` to
`results_real/spatial/<outcome>/` and push them. `field.csv` is the cell-level estimated
field: aggregated, with no ACLED rows.

## Task 5: wait for the next tasks

After Task 4, write "ready for next tasks" in STATUS and push. The cloud session will add
the next tasks to this file: global fits, the space-time model, national indicators and
spatially varying coefficients.
