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
- pull, then `qsub -v CLEAN_OUT=/groups/scastruc/conflict_spatial_work/clean_v2.parquet,ONLY_CLEAN=1 crc/jobs/02_load.sh` (do not overwrite
  `clean.parquet` while jobs read it);
- commit `clean_v2.parquet.report.json` as `results_real/data/clean_v2_report.json` and in STATUS give the row counts against clean.parquet
  and the two new interior fields.
I'll decide from that whether the downstream jobs need a rerun. If the only difference is the first week,
they don't, until the final rerun.

## Reply to STATUS 13:03 (Task 2d) — Task 2e: READY
Thanks. The interior rule works. The 28-43% missing national indicators matter, because the latent models
mean-impute them. Please run on a login or short node, since it only reads parquet:
`python realdata/check_missing.py /groups/scastruc/conflict_spatial_work/clean_v2.parquet results_real/data/missing_v2.json`
Commit the JSON and summarise in STATUS: the missing shares in the Africa+ME box, and which countries account for them.
Keep the running jobs going. The downstream rerun on clean_v2 will happen once, together with any code changes from
the next review round.

## Task 2f: READY (data figure for the paper)
Please pull and run it on a node with enough memory to read clean_v2, for example `qrsh -q long -l mem_free=60G`:
```
python realdata/make_data_figure.py /groups/scastruc/conflict_spatial_work/clean_v2.parquet \
    paper/data/ne_50m_countries.geojson paper/figures/data_events.png
```
(Update: borders now come from the committed Natural Earth file, so no `$COUNTRIES` is needed.) Commit `paper/figures/data_events.png` and `paper/figures/data_events.json`,
which holds aggregate numbers only, and paste the JSON into STATUS.

## Round 3 (after review round 2): READY

The code changed after review round 2 (loader, XGBoost modes, hybrid, comparison and
parametric models). Every real-data result is rerun once, on a new table `clean_r3.parquet`
(`$CLEAN`, set in `env.sh`). The old `clean.parquet` / `clean_v2.parquet` are left alone.

What changed, so you know what to check in the logs:
- **Loader** (`02_load.sh`): each cell-week keeps the whole first row (no mixing of copies); IPC
  phases are averaged over copies and stay missing when no copy has them; the national-indicator
  fallback for country-years without interior cells now works (`nat_source` is `interior`,
  `first_copy` or `missing`); new column `gdp_pc` = World Bank GDP / national population (sum of
  GHSL population of the country's cells), one value per country-year (the loader asserts this).
- **XGBoost** (`03_xgb.sh`, now an array of 8): `original` (orig.parquet, also scored on one row per
  cell-week and on the cell-weeks of the repaired test set), `dedup` (published procedure on the
  repaired table), `corrected` (no class weight, early stopping on 2022 after fitting 2018-2021,
  no current-week climate, log national GDP per capita, IPC missing kept as missing),
  `corrected_te` (location-aware feature computed from 2018-2021 for the early-stopping fit and
  leave-one-year-out over 2018-2022 for the refit). Metrics now include onset rows and scores by test year.
- **Hybrid** (`04_spatial.sh`, `06_global.sh`): offset features as in `corrected`; `diagnostics` in
  `summary.json` and `variogram.csv` are now TEST-period (2023-2024); the training-period ones are in
  `diagnostics_train` and `variogram_train.csv`. The border contrast also splits across-border pairs
  by same/different cross-fitting fold. `summary.json` records `offset_training_region` and the share of
  cells with P(w > 0) above 0.95 / below 0.05. `--land` is now required.
- **Compare** (`05_compare.sh`): adds `xgb_dedup`; corrected models use the raw `logit`; SEs clustered by
  cell and by country; onset / continuation rows; by test year; test-period variograms for every model
  (`compare_<outcome>_variogram.csv`).
- **Parametric models** (`07_glmm.sh`): iid cell effect (nugget) added to every model with a field;
  missingness indicators for national indicators and IPC; climate anomalies with a floored SD
  (precipitation on log1p); national GDP per capita; the central-Sahel space-time comparison now
  fits w on the buffered lattice with the sea barrier (v stays on the observed cells). It also writes
  `test_pred_<outcome>.parquet` (keep on the cluster).

**Before submitting:** if 1693751 (`cs_global`) or 1693752 (`cs_glmm`) from Round 2 are still running,
`qdel` them. They use the old code, their outputs are superseded, and they write to the same
directories as the new jobs.

Submit, from the repo after `git pull`:
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
L=$(qsub -terse -v ONLY_CLEAN=1 crc/jobs/02_load.sh)                       # 1. loader -> $WORK/clean_r3.parquet
X=$(qsub -terse -hold_jid $L crc/jobs/03_xgb.sh | cut -d. -f1)             # 2. xgb, array 1-8
P=$(qsub -terse -hold_jid $L crc/jobs/04_spatial.sh | cut -d. -f1)         # 3. hybrid, Africa+ME
C=$(qsub -terse -hold_jid $X,$P crc/jobs/05_compare.sh | cut -d. -f1)      # 4. compare
G=$(qsub -terse -hold_jid $L crc/jobs/06_global.sh | cut -d. -f1)          # 5. hybrid, world
M=$(qsub -terse -hold_jid $L crc/jobs/07_glmm.sh | cut -d. -f1)            # 6. parametric models
D=$(qsub -terse crc/jobs/10_check_duplicates.sh)                           # independent
echo "load $L xgb $X spatial $P compare $C global $G glmm $M check $D"
```
`ONLY_CLEAN=1` skips rebuilding `orig.parquet`: the `--keep-duplicates` path did not change, so the
existing `$WORK/orig.parquet` is used by the `original` tasks. Log the job ids in STATUS.

After the loader finishes (login or short node; it only reads parquet):
`python realdata/check_missing.py $WORK/clean_r3.parquet results_real/data/missing_r3.json`

Commit (aggregate files only; no parquet):
- `$WORK/clean_r3.parquet.report.json` -> `results_real/data/clean_r3_report.json`; `missing_r3.json`;
  `$WORK/data/check_duplicates.json` -> `results_real/data/check_duplicates.json` (now also per distinct cell-week).
- `$WORK/xgb/metrics_{original,dedup,corrected,corrected_te}_{Conflict,Unrest}.json` (8 files) -> `results_real/xgb/`.
- `$WORK/spatial/<outcome>/{summary.json,variogram.csv,variogram_train.csv,field.csv}` -> `results_real/spatial/<outcome>/`.
- `$WORK/compare/compare_<outcome>.json` and `compare_<outcome>_variogram.csv` -> `results_real/compare/`.
- `$WORK/spatial_global/<outcome>/{summary.json,variogram.csv,variogram_train.csv,field.csv}` -> `results_real/spatial_global/<outcome>/`.
- `$WORK/glmm/{models_<outcome>.json,coefficients_<outcome>.csv,svc_tmax_<outcome>.csv}` -> `results_real/glmm/`.

In STATUS, per job: rc, run time, maxvmem, and any Traceback. For the loader, also give rows/cells against
clean_v2 and `share_rows_nat_from_first_copy`, `share_rows_gdp_pc_missing`. For xgb, the `rounds` of each task.
For glmm, `prep.climate.*.cell_months_sd_zero` and `prep.missing_indicators`. If the `st` part of 07 runs
out of time or memory, rerun it alone with `--models st --st-step 16 --out $WORK/glmm_st` (a separate
directory, so that `models_<outcome>.json` of the other models is not overwritten) and record that.

### Round 3 addendum: figures (run after the spatial jobs and the loader)
```
python paper/make_data_figure.py $CLEAN paper/figures/data_events.png
python paper/make_fitted_figure.py $WORK/spatial/conflict/test_pred.parquet $WORK/spatial/unrest/test_pred.parquet paper/figures/real_fitted.png
python paper/make_real_figures.py
python paper/make_tables.py && python paper/make_sim_figures.py   # after the simulation merge
```
Commit the PNG, JSON and TeX outputs.

## Round 4 (noise or moving structure?): READY
Requested by the PI. There is new code: `realdata/fit_dynamic.py`, and hyperparameter bounds plus an
inner-convergence check in `sc/laplace.py` and `sc/gmrf.py`. Static fits on the test fixture are unchanged.
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
J=$(qsub -terse crc/jobs/11_dynamic.sh | cut -d. -f1)
qsub -hold_jid $J crc/jobs/12_dynamic_compare.sh
```
- Tasks 1-4 (oracle and rolling) should take well under an hour each.
- Tasks 5-8 (space-time) are the expensive ones. On 1-degree v, the latent size is about 3.4k cells x 29 steps
  (13-week steps) or x 15 steps (26-week steps). Please report the wall time and maxvmem of each task.
  If a space-time task looks likely to exceed about 24 h, tell me in STATUS rather than waiting.
- Commit the JSON summaries only, never the parquets:
  - `$WORK/dynamic/<outcome>/summary_*.json` -> `results_real/dynamic/<outcome>/`
  - `variogram_*.csv` and `field_test.csv` -> `results_real/dynamic/<outcome>/`
  - `$WORK/compare/compare_*_st.json` and `compare_*_rolling.json` -> `results_real/compare/`
- Also: the global (1694070) and glmm (1694071) jobs from Round 3. Report them when they finish.

## Round 5 (quarterly updating): READY
Every model is updated each quarter of 2023-2024 with all data before the quarter.
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
X=$(qsub -terse -t 7-8 crc/jobs/03_xgb.sh | cut -d. -f1)       # corrected_te again: adds the logit_upd column
U=$(qsub -terse crc/jobs/13_update.sh | cut -d. -f1)            # latent models, about 3 h (space-time fit)
qsub -hold_jid $X,$U crc/jobs/14_update_compare.sh
```
- Commit these:
  - `results_real/xgb/metrics_corrected_te_*.json`, which now has a `quarterly_update` block;
  - `results_real/dynamic/<outcome>/summary_update.json`;
  - `results_real/compare/compare_*_update.json` and the `_variogram.csv` files.
- Not done yet: the rolling-compare label you flagged. The rows are 2024 only and the label is wrong. I'll fix it in
  compare.py later; no rerun is needed.

## Round 6 (bivariate model: S5 simulation): READY
New code: `CopyBlock` in `sc/gmrf.py` (a shared latent component with a loading) and `sims/s5_bivariate.py`.
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
J=$(qsub -terse crc/jobs/15_s5.sh | cut -d. -f1)
# when the array is done (hold or run by hand):
qsub -hold_jid $J -N cs_s5merge -q long -l mem_free=16G -cwd -j y -o /groups/scastruc/conflict_spatial_work/logs/ \
    -b y "source crc/jobs/env.sh && cd sims && N_JOBS=1 python s5_bivariate.py"
```
- Each replicate takes about 10 min; each task has 2.
- Commit `results/s5_raw.csv` and `results/s5_summary.csv`, and paste the summary table into STATUS.

## Round 7 (bivariate model on the data): READY
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
qsub crc/jobs/16_bivariate.sh
```
- Task 1 (West Africa and Sahel) should take a few hours.
- Task 2 (all of Africa + ME) is large: about 7M stacked rows and 12 hyperparameters.
  - Please report its maxvmem after the first hour.
  - If it looks like it will exceed about 48 h, say so in STATUS. I may then switch it to sub-regions.
- Commit `$WORK/bivariate/<region>/summary_biv.json` to `results_real/bivariate/<region>/`.
- Paste into STATUS the `lag_coefficients` and `hyper` of each model.

## Round 7b (faster engine): READY
`sc/laplace.py` now has three speed-ups. All give identical results on the test problems (same log-ML,
coefficients and hyperparameters).
1. The Hessian A'WA is assembled blockwise, with dense algebra for the covariate columns. That is about 18x faster at
   1M rows, and it is the main cost for the big tables.
2. Parallel finite-difference gradients (`FIT_JOBS` processes): 2.5x on 4 cores for S5, and more with 12 hyperparameters.
3. Optional CHOLMOD backend, used automatically if scikit-sparse is installed:
   `conda install -c conda-forge scikit-sparse` in the conflict env. It is not required, and the speed is similar
   at these sizes.

Please:
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
qdel 1696206.2            # africa_me bivariate on the old engine
qsub -t 2 crc/jobs/16_bivariate.sh   # 12 slots now
```
Report the wall time.

## CRC resource limits (from the CRC documentation; replaces the "4 cores per job" rule)
The PI asked to follow the documented limits. From https://docs.crc.nd.edu/infrastructure/crc_uge_env.html
and https://docs.crc.nd.edu/new_user/quick_start.html:
- long queue: 15 days (the quick start says 14), **no limit on the number of cores**;
- `-pe smp`: up to 64 cores on a single machine; general-access long nodes have 64 cores and 256 GB;
- per user: at most **50 running jobs** across all queues, and at most 2,000 tasks per array job;
- rule of thumb: request **1 core per 4 GB of RAM** the job needs, so the memory is reserved for it;
- requests of 4-8 cores may start sooner; that is about queue wait, not a limit.

What to do:
- **Leave 1699951.2 running** (africa_me bivariate, 4 slots). It is several hours in.
- Future heavy latent-model fits use `-pe smp 16` with `FIT_JOBS=8` (8 gradient workers with 2 BLAS threads each).
  `16_bivariate.sh` is updated.
- For any other job whose memory needs more cores than it requests under the 4 GB rule, raise `-pe smp` to match.

## Round 8 (review round 3: unrest-specific episodes): READY, priority before Sunday night
The joint model may hide real escalation when unrest has episodes of its own. Two runs test this.
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
J=$(qsub -terse crc/jobs/15_s5.sh | cut -d. -f1)          # S5 v2: 40 tasks, 4 slots each, about 1 h per task
qsub -hold_jid $J crc/jobs/09_s5merge.sh  # or the same merge wrapper you used in Round 6 (cd sims && N_JOBS=1 python s5_bivariate.py)
qsub -t 1 crc/jobs/17_biv2.sh                              # West Africa biv2, 16 slots: PRIORITY
```
- Leave 1699951.2 (africa_me biv) running. Run `17_biv2.sh -t 2` (africa_me biv2) only after it finishes.
- Commit:
  - `results/s5_raw.csv` and `results/s5_summary.csv` (now with 4 scenarios and biv_st2);
  - `results_real/bivariate/west_sahel/summary_biv2.json`.
- Paste the two summaries into STATUS.

## Round 9 (joint and conditional prediction): READY
The PI wants interpretation AND joint and conditional prediction in both the simulations and the application.
```
cd /users/scastruc/projects/26.Castruccio.conflict.unpublished && git pull
qsub crc/jobs/18_joint_pred.sh        # tasks 1-2
J=$(qsub -terse crc/jobs/15_s5.sh | cut -d. -f1)   # S5 v3: new checkpoint s5v3, 40 tasks
# then the same S5 merge wrapper as before, held on $J
```
- **Joint prediction.** `18_joint_pred.sh` reuses the saved hyperparameters (`--fixed-theta summary_biv.json`).
  - Only the univariate unrest models are optimised, so it should take about 1 h, not 7.
  - Commit `results_real/bivariate/<region>/summary_jointpred.json`.
- **S5 v3.** Each replicate adds a univariate unrest model and the joint and conditional scores.
  - Commit `results/s5_raw.csv`, `s5_summary.csv` and the new `s5_prediction.csv`.
- **biv2 jobs.** Leave the running jobs (1701825, 1705023) as they are; nothing here changes them.
