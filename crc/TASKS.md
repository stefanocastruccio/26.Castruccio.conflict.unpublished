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
3. Two inputs that Luca's pipeline used are not on the CRC. Download both (they are public)
   into `$WORK/inputs/`; both are **required**:
   - Natural Earth 1:10m Admin 0 Countries: done, `$WORK/world_map/ne_10m_admin_0_countries.shp`
     (`$COUNTRIES` in `env.sh`)
   - World Bank GDP, current US$ (indicator `NY.GDP.MKTP.CD`), CSV:
     `$WORK/inputs/GDP.csv`
     (from https://data.worldbank.org/indicator/NY.GDP.MKTP.CD, "Download CSV": take the
     `API_NY.GDP.MKTP.CD_DS2_*.csv` file from the zip; the loader skips its metadata lines).
     Check it has columns `Country Name`, `Country Code` and `2018` ... `2024`.
4. Report in STATUS: versions, the CUDA flag, the two input paths, and the GDP file's
   header line.

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

---

## Reply to STATUS 10:50 (row counts)

Yes, the drop from 34.7M to 21.7M rows is intended. The loader removes cells whose biome is `N/A`
(rock and ice, mostly Antarctica and Greenland), as Luca's `model_xgb.py` also does
(`data[data.biome_str != 'N/A']`). That takes 94,730 cells down to 59,344. `orig.parquet`
uses the same filter, so the two tables cover the same cells. Thanks; the Task 2-4 results
are received. The next tasks will be added below.

---

## Round 2 (added 2026-11-06, approved by Stefano): ON HOLD, do not start until this line says READY

Run `git pull` first. Same rules as before: summaries only, nothing over a few MB, and no
parquet or prediction files in git.

### Task 6: like-for-like comparison (Africa + Middle East)

`fit_acled.py` now also writes `test_pred.parquet` (test-period predictions; keep it on the
CRC). Then:

1. Rerun `crc/jobs/04_spatial.sh`, which takes about 5 minutes.
2. Submit `crc/jobs/05_compare.sh` held on that rerun
   (`qsub -hold_jid <id> crc/jobs/05_compare.sh`).
3. Copy `$WORK/compare/compare_{conflict,unrest}.json` to `results_real/compare/`.
4. In STATUS, report for each outcome the log score and AUC-PR of `xgb_corrected`,
   `xgb_crossfit` and `hybrid`, plus `logscore_diff_hybrid_minus_xgb_corrected` (mean and SE).

### Task 7: global hybrid model (`crc/jobs/06_global.sh`)

This is the same model as Task 4 on the whole world: about 59k cells, 15M training rows.
It may take hours. Record maxvmem from `qstat -j` while it runs.

- Copy `summary.json`, `variogram.csv` and `field.csv` to `results_real/spatial_global/<outcome>/`.
- Do not copy `test_pred.parquet` or `offset.npy`.

### Task 8: parametric models (`crc/jobs/07_glmm.sh`, Africa + Middle East)

This fits `realdata/fit_glmm.py`: glm, country, field, spatialplus and svc, plus the
space-time comparison on the Sahel / Horn of Africa box. It prints progress after each
model.

- Copy `models_<outcome>.json`, `coefficients_<outcome>.csv` and `svc_tmax_<outcome>.csv`
  to `results_real/glmm/`.
- If the `st` part runs past 48 h or runs out of memory, rerun with `--models st --st-step 16`
  and log that you did.

Tasks 6, 7 and 8 are independent and can run at the same time.
