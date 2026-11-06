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

## Round 2, revised after the internal review: READY (start now)

Run `git pull` first. The pipeline changed after an internal review: ACLED coverage,
country-grouped cross-fitting with the corrected model's features, a sea mask for the
barrier, a fixed persistence baseline, location-aware baselines and new parametric-model
covariates. **All Round 1 tables must be recomputed.** The same rules apply as before:
summaries only, and nothing large or row-level in git.

### Task 1b: ACLED coverage dates (front-end, light)

Try to obtain ACLED's official "country and time-period coverage" table from acleddata.com
(it is usually an XLSX on the data-coverage / knowledge-base pages).

- If you get it, write `$WORK/inputs/acled_coverage.csv` with columns `ISO_A3,start`
  (ISO alpha-3 code and the date coverage starts, YYYY-MM-DD) for every country, and commit
  a copy as `results_real/data/acled_coverage.csv`. It is public metadata, not event data.
- If you can't get it, skip this step. The loader then estimates coverage from the data;
  record in STATUS that the estimated rule was used.

### Task 2r: rebuild the tables (`02_load.sh`)

- Copy `$WORK/clean.parquet.report.json` to `results_real/data/clean_report.json`. It holds
  the counts for duplicates, the coverage drop, sea-centroid cells and the national-indicator
  fallbacks.
- `orig.parquet` is rebuilt the same way as before.

### Task 2c: direct checks on the original files (`10_check_duplicates.sh`, independent)

Copy `$WORK/data/check_duplicates.json` to `results_real/data/check_duplicates.json`. It holds
aggregate shares only.

### Task 3r: XGBoost (`03_xgb.sh`, now an array of 6, held on 2r)

The array covers original, corrected, and corrected with target encoding, for both outcomes.
Copy the six `metrics_*.json` files to `results_real/xgb/`, replacing the old files.

### Task 4r: hybrid models, Africa and Middle East (`04_spatial.sh`, held on 2r)

These now use `--land "$COUNTRIES"`. Copy `summary.json`, `variogram.csv` and `field.csv` to
`results_real/spatial/<outcome>/`.

### Task 6: like-for-like comparison (`05_compare.sh`, held on 3r and 4r)

Copy `compare_{conflict,unrest}.json` to `results_real/compare/`.

### Task 7: global hybrid models (`06_global.sh`, held on 2r)

This may take hours. Record maxvmem. Copy `summary.json`, `variogram.csv` and `field.csv` to
`results_real/spatial_global/<outcome>/`.

### Task 8: parametric models (`07_glmm.sh`, held on 2r)

Copy `models_<outcome>.json`, `coefficients_<outcome>.csv` and `svc_tmax_<outcome>.csv` to
`results_real/glmm/`. If the `st` part goes past 48 h or runs out of memory, rerun it alone
with `--models st --st-step 16` and record that in STATUS.

### Task 9: simulation studies (`08_sims.sh` array 1-40, then `09_sims_merge.sh`)

The simulations were revised too. Fields are now drawn from the exact Matern process,
S0 has a rare-event setting, S1 has location-aware baselines, S2 uses 48 blocks with
B = 40, S3 has a mixed scenario and weekly space-time fits, and S4 has raw temperature
and interaction-credited SHAP slopes. They need no data and can start immediately.

- `qsub crc/jobs/08_sims.sh`, then `qsub -hold_jid <id> crc/jobs/09_sims_merge.sh`.
- Commit and push the updated `results/*.csv`, `results/s1_maps.png` and
  `paper/tables/*.tex`. Leave out `results/*_ckpt*.jsonl`, which are gitignored.

Submit order: 2r, 2c and 9 now; then (3r, 4r, 7, 8) held on 2r; then 6 held on 3r and 4r.
Log the job ids.

## Reply to STATUS (Task 2r report), 2026-11-06 — Task 2d: READY
Thanks, both points were real issues in the code or the report:
1. **Interior-cell report.** The old `countries_without_interior_cell` listed any country with at least one
   country-year lacking interior values. Most likely that is every country, because the national
   indicators are not published for the last year or two (all NaN), not because the interior rule failed. The
   loader now also reports `countries_without_any_interior_cell`, `share_rows_nat_from_interior_by_year`
   and `share_rows_nat_missing_by_year`.
2. **Coverage edge.** Fixed: a week counts as covered if it *ends* on or after the coverage start, so the
   first week (which starts before 1 January 2018) is no longer dropped for countries covered from 2018.

**Do not cancel the running jobs.** Please:
- pull, then `qsub -v CLEAN_OUT=$WORK/clean_v2.parquet,ONLY_CLEAN=1 crc/jobs/02_load.sh` (do not overwrite
  `clean.parquet` while jobs read it);
- commit `results_real/data/clean_v2_report.json` and in STATUS give the row counts against clean.parquet
  and the two new interior fields.
I'll decide from that whether the downstream jobs need a rerun. If the only difference is the first week,
they don't, until the final rerun.
