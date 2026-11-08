# CRC status log

The CRC session appends entries here (newest at the bottom): date/time, task, job ids,
exit status, key numbers, and any error output.

---

## 2026-11-06 09:52 EST: Task 1 done; Tasks 2-4 submitted

**Task 1 (environment).** Built a dedicated env, `/users/scastruc/envs/conflict` (batch job 1693177), so that the
`geo` env, which a running geostatistics batch is using, stays untouched.
- Versions: Python 3.12, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1,
  pyarrow 26.0.1, pymetis 2025.2.2, gpboost 1.8.0, geopandas 1.2.0, pyogrio 0.13.0, shapely 2.2.0.
- `xgboost.build_info()['USE_CUDA']` is True (the pip wheel is CUDA-enabled; the GPU itself is tested in Task 3).

**Input files.** Neither expected input exists. `$DATA` (`/groups/scastruc/lmenical/drew/data`) contains only
`partitions/`, and neither file turned up anywhere else under `/groups/scastruc` or `/users/scastruc`.
- `$DATA/world_map/ne_10m_admin_0_countries.shp`: **missing**. I downloaded the public Natural Earth 10m
  admin-0 countries file (v5.1.x) to `$WORK/world_map/` and pointed `$COUNTRIES` at it in `env.sh`.
- `$DATA/socioeconomic/GDP.csv`: **missing**. `02_load.sh` now passes `--gdp-csv` only if the file exists,
  so the clean table keeps the loader's "first copy" GDP. Push or point me to a GDP.csv if you want the
  GDP repair, and I'll rerun Task 2.

**Header changes in `crc/jobs/`.**
- Absolute path to `env.sh`, because SGE runs a spooled copy of the script.
- `module load conda` plus the `conflict` env; threads set to `$NSLOTS`.
- `h_rt`: 24 h (load), 48 h (xgb; the gpu queue allows ≤96 h), 96 h (spatial; largemem allows ≤168 h).
- Logs go to `$WORK/logs/`.
- `-l gpu_card=1` is correct on this cluster, so it is unchanged.

**Jobs submitted.**
- 1693181 `cs_load` (largemem).
- 1693182 `cs_xgb` (gpu, array 1-4) and 1693183 `cs_spatial` (largemem, array 1-2), both held until `cs_load`
  finishes.

## 2026-11-06 10:00 EST: jobs cancelled; required inputs now in place

**Jobs cancelled.** I deleted 1693181 `cs_load`, 1693182 `cs_xgb` and 1693183 `cs_spatial` while they were still queued,
before any of them ran, because they had been submitted without GDP.csv.

**Inputs.**
- Countries: `/groups/scastruc/conflict_spatial_work/inputs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp`
  (Natural Earth 10m admin-0, VERSION 5.1.1, fetched from naciscdn.org).
- GDP: `/groups/scastruc/conflict_spatial_work/inputs/GDP.csv`. This is `API_NY.GDP.MKTP.CD_DS2_en_csv_v2_512.csv`
  from `https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv`, unmodified:
  271 lines, "Last Updated Date" 2026-10-08, data for 1960-2025.
- GDP header line (line 5):
  `"Country Name","Country Code","Indicator Name","Indicator Code","1960","1961",...,"2024","2025",`
  The first 4 lines are metadata (Data Source, blank, Last Updated Date, blank).

**Next.** Your updated `crc/jobs/02_load.sh`, with the `$WORK/inputs` paths and the stop when an input is missing, is
not on origin/main yet; the last commit there is 91345bf. I'll resubmit Tasks 2-4 once it is pushed.

## 2026-11-06 09:53 EST: Tasks 2-4 resubmitted with the required inputs

- Pulled 988ff5c (02_load.sh now requires both inputs).
- `env.sh`: `COUNTRIES` now defaults to `$WORK/inputs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp`.
  `GDP_CSV` defaults to `$WORK/inputs/GDP.csv`, as set in 02_load.sh.
- Jobs:
  - 1693184 `cs_load` (largemem).
  - 1693185 `cs_xgb` (gpu, array 1-4) and 1693186 `cs_spatial` (largemem, array 1-2), both held on 1693184.
- The note in my previous entry, that 988ff5c wasn't on origin yet, is out of date: it arrived while I was pushing.

## 2026-11-06 10:35 EST: cs_load moved from largemem to long; now running
- **largemem is closed to us:** the queue has an access list (`largemem`, `crc`) that doesn't include scastruc, so
  cs_load sat in qw for 40 min. I moved 1693184 `cs_load` and 1693186 `cs_spatial` to `-q long -l mem_free=120G`
  with `qalter`. Long-queue nodes are mostly 64-core with 251 GB RAM.
- The headers of `02_load.sh` and `04_spatial.sh` are changed the same way.
- 1693184 started at 10:33:47 on d32cepyc192. 1693185 `cs_xgb` (gpu) and 1693186 are held until it finishes.

## 2026-11-06 10:50 EST: Task 2 done (job 1693184, rc 0); Tasks 3-4 running
Job 1693184 `cs_load` ran on d32cepyc192 (long) from 10:33:50 to 10:47, about 14 min. Loader output:
```
input rows 70,033,444; distinct cell-weeks 34,671,180; duplicated share of rows 50.5%
wrote /groups/scastruc/conflict_spatial_work/clean.parquet: 21,719,904 rows, 59,344 cells, 366 weeks, 47 columns
input rows 70,033,444; distinct cell-weeks 34,671,180; duplicated share of rows 50.5%
wrote /groups/scastruc/conflict_spatial_work/orig.parquet: 55,474,944 rows, 59,344 cells, 366 weeks, 46 columns
```
- File sizes: clean.parquet is 1.02 GB and orig.parquet is 1.60 GB.
- **Peak memory: not available.** `qacct` doesn't work on crcfe01 (`/opt/sge/crc/common/accounting: No such file`).
  For Tasks 3-4 I'll record maxvmem from `qstat -j` while the jobs are running.
- Please check: clean has 21.7M rows, but the loader reports 34.7M distinct cell-weeks. The difference (and the
  59,344 cells against roughly 95k grid_id per partition earlier) suggests the loader also filters cells, e.g. to land
  or to cells with a country assignment. Can you confirm that's intended?
- Tasks 3-4 started at 10:47:53: 1693185 `cs_xgb` tasks 1-4 on gpu@qa-a10-015 (A10), and 1693186 `cs_spatial`
  tasks 1-2 on long@d32cepyc192/105.

## 2026-11-06 10:52 EST: Task 3 done (job 1693185, 4 tasks, rc 0); Task 4 running
- Job 1693185 `cs_xgb` ran on gpu@qa-a10-015 (NVIDIA A10), 10:47 to 10:51: the corrected tasks took about 2 min each,
  the original ones about 4 min. No Tracebacks.
- The four metrics JSON files are in `results_real/xgb/`. The prediction parquets stay in `$WORK/xgb`.

| mode | outcome | auc_pr | auc_pr_border_cells | auc_pr_interior_cells | persistence_auc_pr | rounds |
|---|---|---|---|---|---|---|
| corrected | Conflict | 0.5485 | 0.5882 | 0.5279 | 0.5108 | 858 |
| corrected | Unrest | 0.5527 | 0.5423 | 0.5552 | 0.5001 | 203 |
| original | Conflict | 0.9342 | 0.9665 | 0.4976 | 0.8896 | 1998 |
| original | Unrest | 0.9175 | 0.9684 | 0.5281 | 0.8519 | 1474 |

- In the original (duplicated-row) analysis, AUC-PR is about 0.93 overall, but it comes almost entirely from border
  cells (about 0.97); interior cells give 0.50-0.53. Persistence alone reaches 0.85-0.89. After correction, AUC-PR
  falls to about 0.55 and is roughly equal for border and interior cells. That is consistent with the
  same-week leakage through `*_rollsum_4` described in STUDENT_CODE_REVIEW.md item 1b.
- Task 4: job 1693186 `cs_spatial` tasks 1-2 are running on long. They are in hyperparameter optimisation; maxvmem so far is maxvmem=16.059G .

## 2026-11-06 10:55 EST: Task 4 done (job 1693186, 2 tasks, rc 0)
Job 1693186 `cs_spatial` ran on long (d32cepyc192 and 105) from 10:47:53 to about 10:53, about 5 min. maxvmem was 16.1 GB.
The outputs are in `results_real/spatial/{conflict,unrest}/`: `summary.json`, `variogram.csv`, and `field.csv` (19,416 cells).
`offset.npy` stays in `$WORK`.

| outcome | model | log score | Brier | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|
| conflict | XGB cross-fit | 0.0630 | 0.01608 | 0.508 | 0.942 |
| conflict | XGB + field | 0.0600 | 0.01595 | 0.527 | 0.958 |
| unrest | XGB cross-fit | 0.0672 | 0.01747 | 0.522 | 0.946 |
| unrest | XGB + field | 0.0633 | 0.01698 | 0.548 | 0.959 |

- Hyperparameters:
  - conflict: country sd 0.354, field range 277 km, field sd 1.24.
  - unrest: country sd 0.586, field range 72 km, field sd 1.11.
- Persistence AUC-PR: 0.228 for conflict, 0.233 for unrest.
- Residual diagnostics:
  - Moran's I: 0.452 for conflict, 0.110 for unrest (p = 0.005 for both).
  - Neighbour residual correlation within country vs across a border: 0.482 vs 0.305 for conflict, and 0.114 vs 0.108
    for unrest (24,508 within-country pairs and 2,040 across-border pairs).

## 2026-11-06 10:55 EST: Task 5: ready for next tasks

## 2026-11-06 12:40 EST: Round 2 submitted

### Task 1b: ACLED coverage, from the official table
- Source: ACLED's "Country and Time-Period coverage" PDF, updated February 2022
  (https://acleddata.com/acleddatanew/wp-content/uploads/dlm_uploads/2019/01/ACLED_Country-and-Time-Period-coverage_updatedFeb2022.pdf).
  The knowledge-base page itself has no XLSX download. No newer table was available.
- I transcribed all 243 country rows by hand into `ISO_A3,start`, using the first day of the start month. Examples:
  Africa 1997-01-01 (South Sudan 2011), Middle East 2015-2017, South and Southeast Asia 2010-2020,
  Europe and Central Asia 2018 or 2020, the Americas 2018 (USA 2020, Canada 2021), Oceania 2021.
- The loader drops any country missing from the file, so I checked the codes against Natural Earth
  (`ISO_A3`, falling back to `ADM0_A3` when it is -99) in a short batch job. Natural Earth has 24 codes that are not in
  ACLED's table. Each got the coverage start of the country or region ACLED files them under:
  - ESB, WSB, CYN, CNM -> Cyprus (2018).
  - SOL -> Somalia (1997).
  - ESH -> Morocco/Africa (1997).
  - BRT -> Egypt/Sudan (1997).
  - USG -> Cuba (2018).
  - BRI, SPI, BJN, SER -> 2018 (Americas).
  - HKG, MAC -> China (2018).
  - KAS -> India (2016, the later of India and Pakistan).
  - KAB -> Kazakhstan (2018).
  - PGA, SCR -> Philippines (2016).
  - ALA -> Finland (2020).
  - ATF, IOA, CSI, ATC, CLP -> 2021.
- Caveat: Natural Earth's FRA polygon includes the overseas departments (Guadeloupe, Martinique, French Guiana,
  Reunion, Mayotte). Those cells therefore get France's start, 2020, rather than ACLED's 2018 (Americas) or 2020 (Africa).
- The file has 267 rows: `$WORK/inputs/acled_coverage.csv`, committed as `results_real/data/acled_coverage.csv`.

### Jobs
- Task 9: 1693743 `cs_sims` (array 1-40, all running), then 1693744 `cs_sims_merge` (held on it).
- Task 2r: 1693748 `cs_load` (long, mem_free=120G; running since 12:38).
- Held on 2r: 1693749 `cs_xgb` (gpu, 1-6), 1693750 `cs_spatial` (1-2), 1693751 `cs_global` (1-2) and 1693752 `cs_glmm` (1-2).
- Task 6: 1693753 `cs_compare` (1-2), held on 1693749 and 1693750.

## 2026-11-06 12:41 EST: pulled af03a94; Task 2c submitted
- Task 2c: job **1693762** `cs_check` (long, mem_free=120G).
- Pulled af03a94 before any of the held jobs (3r, 4r, 6, 7, 8) started, so they will run the new code.
- Two jobs were already running when af03a94 arrived:
  - 1693748 `cs_load` (2r) started at 12:38. 02_load.sh and load_partitions.py are unchanged in af03a94, so this
    doesn't matter.
  - 1693743 `cs_sims` (Task 9, 40 tasks) also started at 12:38, on the previous `sc/laplace.py`. The only change
    there is batched sampling in `LatentGaussianModel.sample` (batch=25). It draws from the same distribution, but the
    RNG stream differs when n > 25. I let the simulations continue rather than restart them. If you want them rerun on
    af03a94, say so and I'll qdel them, delete the shard checkpoints and resubmit.

## 2026-11-06 12:46 EST: Task 2c done (job 1693762, rc 0, about 3 min)
`results_real/data/check_duplicates.json` is committed. In brief:
- **Duplicated cell-weeks:** 2,946,558. Copies most often differ in EPI_index (62%), v2x_libdem (62%), gdp (49%),
  GHG/MF/ECO (39%), IPC phase_1-3 (about 25%), and the climate variables (7-11%).
- **Leakage of the current week into the original 4-week counts:**
  - About 22% of event-weeks leak, for both Conflict and Unrest.
  - The original count differs from the true count in 4.4% of rows (Conflict) and 4.9% (Unrest), out of 69.4M rows.
  - AUC-PR of the original count is 0.883 against 0.495 for the true count (Conflict), and 0.833 against 0.450 (Unrest).

## 2026-11-06 12:52 EST: Task 2r done (job 1693748, rc 0, 12:38-12:50); 3r, 4r, 7 and 8 running
```
input rows 70,033,444; distinct cell-weeks 34,671,180; duplicated share of rows 50.5%
wrote clean.parquet: 19,321,887 rows, 59,344 cells, 366 weeks, 48 columns   (Round 1 had 21,719,904)
wrote orig.parquet: 55,474,944 rows, 59,344 cells, 366 weeks, 46 columns
```
- `results_real/data/clean_report.json` is committed. maxvmem was 24.5 GB.
- Report contents:
  - `coverage_source = acled_coverage.csv`.
  - Cell-weeks dropped before coverage: 7,156,461 across 160 countries. The largest are ATA (4.02M), CAN (1.27M),
    USA (0.52M), AUS (0.48M) and GRL (0.32M).
  - 9,732 sea-centroid cells were assigned to the nearest country. GDP is "reapportioned".
- **Please check two things in the report:**
  1. `countries_without_interior_cell` lists **243 countries**, which is nearly every country (AFG, AGO, BRA, CHN, ...).
     `cells_nat_from_first_copy` is **94,730**, which is more than the 59,344 cells in the output. Taken together,
     these suggest the interior-cell rule never matched, so national indicators come from the first copy everywhere.
     That could be a bug in how interior cells are detected.
  2. Countries whose coverage starts on 2018-01-01 still lose a few cell-weeks: ARG 1,170, BRA 2,929, CHN 3,898,
     RUS 12,779. Presumably that's the first `week_start`, which falls before 2018-01-01.
- I have not stopped anything. 1693749 `cs_xgb` (6 tasks), 1693750 `cs_spatial`, 1693751 `cs_global` and
  1693752 `cs_glmm` are all running on this table. Tell me in TASKS.md if you want them cancelled and rerun.

## 2026-11-06 12:53 EST: Task 2d submitted
- Pulled 7da60eb. Job **1693835** `cs_load` runs with `-v CLEAN_OUT=/groups/scastruc/conflict_spatial_work/clean_v2.parquet,ONLY_CLEAN=1`.
- clean.parquet is not touched, and the running jobs (3r, 4r, 7, 8, Task 9) are left alone.

## 2026-11-06 12:55 EST: Task 3r done (job 1693749, 6 tasks, gpu, 12:50-12:53, no Tracebacks; maxvmem about 69 GB)
The six `metrics_*.json` files replace the old ones in `results_real/xgb/`. They were computed on clean.parquet from 2r, not on clean_v2.

| file | auc_pr | auc_pr_border_cells | auc_pr_interior_cells | persistence_auc_pr | auc_roc | rounds | n_test_rows |
|---|---|---|---|---|---|---|---|
| corrected_Conflict | 0.5445 | 0.5881 | 0.5218 | 0.5109 | 0.9651 | 810 | 4787224 |
| corrected_Unrest | 0.5569 | 0.5468 | 0.5594 | 0.5003 | 0.9587 | 879 | 4787224 |
| corrected_te_Conflict | 0.5637 | 0.5984 | 0.5461 | 0.5109 | 0.9726 | 195 | 4787224 |
| corrected_te_Unrest | 0.5827 | 0.5672 | 0.5864 | 0.5003 | 0.9663 | 147 | 4787224 |
| original_Conflict | 0.9342 | 0.9665 | 0.4976 | 0.8896 | 0.9945 | 1998 | 14425130 |
| original_Unrest | 0.9175 | 0.9684 | 0.5281 | 0.8519 | 0.9926 | 1474 | 14425130 |


## 2026-11-06 12:58 EST: Task 9 done
- 1693743 `cs_sims`: all 40 tasks finished between 12:38 and 12:56, with about 15 GB maxvmem and no Tracebacks.
  They ran on the `sc/laplace.py` before af03a94; see the 12:41 entry.
- 1693744 `cs_sims_merge`: no FAILED scripts. It wrote s0, s1_scores, s1_field, s2, s3, s4, s4_extra and barrier `.tex`.
- Committed: `results/*.csv` (15 files changed), `results/s1_maps.png` and `paper/tables/*.tex` (7 changed;
  barrier.tex is unchanged). The `*_ckpt*.jsonl` files are not committed.

## 2026-11-06 13:03 EST: Task 2d done (job 1693835, rc 0, 12:53-13:01, maxvmem about 25 GB)
`results_real/data/clean_v2_report.json` is committed. clean_v2.parquet compared with clean.parquet (2r):

| | clean (2r) | clean_v2 (2d) |
|---|---|---|
| rows | 19,321,887 | **19,365,684** (+43,797) |
| cells / weeks | 59,344 / 366 | 59,344 / 366 |
| cell-weeks dropped before coverage | 7,156,461 | 7,078,822 (-77,639) |
| countries with dropped weeks | 160 | 79 |
| sea-centroid cells | 9,732 | 9,732 |

- The extra rows are the first week. The largest gains are ATA 25,467, RUS 12,779, CAN 8,057, USA 4,917, CHN 3,898,
  AUS 3,056, GRL 3,007 and BRA 2,929 cell-weeks.
- **`countries_without_any_interior_cell`: [] (0 countries).**
- **`share_rows_nat_from_interior_by_year`:** 2017: 0.000; every year from 2018 to 2024: 0.729.
- `share_rows_nat_missing_by_year`:
  - In 2017 (the single week starting in Dec 2017) every indicator is 1.0 missing.
  - In 2018-2024 the missing shares are roughly constant: EPI_index 0.276, v2x_libdem 0.311, and
    GHG/MF/ECO 0.41-0.43.
  - So the gaps are not concentrated in the last year or two. About 28-43% of rows lack these indicators in every
    year, probably countries that the indicator sources don't cover.
- `cells_nat_from_first_copy_any_year`: 94,730. That is all input grid_ids, including cells later dropped, which is
  why it's more than 59,344.

## 2026-11-06 13:10 EST: Task 2e done (job 1693848, rc 0, about 5 min)
I ran it as a short batch job (long queue, 4 cores, mem_free=60G) rather than on the 1-core login node, because it reads the 0.9 GB parquet.
`results_real/data/missing_v2.json` is committed.

**Share of rows missing in the Africa + ME box (4,976,634 rows):**
- Sustainability, GHG_pc_t / MF_pc_t / ECO_pc_gha: **0.279**. Most of it comes from COD 277,794, IRN 226,920,
  LBY 216,672, SAU 213,520, TUR 118,584, SSD 73,932, SDN 65,940, CIV 40,626, COG 40,260 and SYR 25,986 cell-weeks.
  These are whole countries absent from the sustainability source.
- gdp: 0.036. SSD 73,932, YEM 48,515, SOL 21,228, ERI 14,274, ESH 11,712, SYR 7,384, KOS 2,196.
- EPI_index and v2x_libdem: 0.004 each. KOS 2,196, GMB 1,098, SWZ 1,050, DZA 854, COD 759, PSE 732, ...
- electricity_access: 0.005. MAR 21,228, ESH 5,490, SDN 366.
- Other covariates in the box:
  - IPC phase_1-5: 0.627.
  - pm25 and its 4-week mean: 0.555 / 0.560.
  - Event rollsums: 0.011, the warm-up weeks after coverage starts.
  - Climate rollmeans: 0.011.

**Whole table (19,365,684 rows):**
- GHG/MF/ECO: 0.178, mostly USA 1.14M, KAZ 0.41M, COD, IRN, LBY, SAU and GRL.
- v2x_libdem: 0.013 (GRL 168,606, ATA, RUS, ...).
- EPI_index: 0.004. gdp: 0.014.
- IPC phase: 0.896. pm25: 0.556. ppt: 0.208. tmin/tmax/hi/wbgt: 0.053.

**Note.** These shares are much lower than the report's `share_rows_nat_missing_by_year`
(EPI 0.276, v2x 0.311, GHG 0.41-0.43). That suggests the report field is computed before the per-country fill or
before the coverage and land filtering, so it doesn't describe the final table. In the final table, only the
sustainability trio (about 28% in the box) and IPC/pm25 have large gaps.

## 2026-11-06 13:11 EST: Task 4r done (job 1693750, 2 tasks, rc 0, no Tracebacks; maxvmem about 15.7 GB)
- Run on clean.parquet (2r) with `--land`. Each outcome has 4,975,444 rows, 13,719 cells and 3,725 barrier
  (sea) cells, with country folds.
- `summary.json`, `variogram.csv` and `field.csv` replace the old files in `results_real/spatial/{conflict,unrest}/`.
- Task 6 (1693753 `cs_compare`) started as soon as this finished.

| outcome | model | AUC-PR | AUC-ROC | log score | hyperparameters |
|---|---|---|---|---|---|
| conflict | XGB cross-fit (country folds) | 0.444 | 0.916 | 0.0716 | |
| conflict | + barrier field | 0.521 | 0.959 | 0.0605 | country sd 0.80, range 264 km, field sd 1.47 |
| conflict | + iid cell | 0.521 | 0.958 | 0.0609 | country sd 1.40, cell sd 1.30 |
| unrest | XGB cross-fit | 0.507 | 0.939 | 0.0697 | |
| unrest | + barrier field | 0.547 | 0.959 | 0.0634 | country sd 1.05, range 68 km, field sd 1.20 |
| unrest | + iid cell | 0.547 | 0.959 | 0.0634 | country sd 1.10, cell sd 1.11 |

- Persistence AUC-PR: 0.446 for conflict and 0.451 for unrest (the fixed baseline).
- Moran's I of the residuals: 0.620 for conflict, 0.210 for unrest.
- Within-country vs across-border correlation: 0.644 vs 0.417 for conflict, 0.232 vs 0.109 for unrest.

## 2026-11-06 13:14 EST: Task 2f done (job 1693864: long, 4 cores, mem_free=60G, rc 0, about 2 min)
Committed `paper/figures/data_events.png` and `paper/figures/data_events.json`. I ran it with qsub rather than qrsh.

`data_events.json`:
```json
{
 "rows": 19365684,
 "cells": 59344,
 "weeks": 366,
 "Conflict": {
  "overall_share_event_weeks": 0.012553132644320748,
  "share_cells_any_event": 0.18028781342679967,
  "share_cells_event_in_half_of_weeks": 0.004263278511728229,
  "top1pct_cells_share_of_event_weeks": 0.4645043192102016,
  "weekly_share_min": 0.003560504683616899,
  "weekly_share_max": 0.01608201108774613
 },
 "Unrest": {
  "overall_share_event_weeks": 0.021250475841700193,
  "share_cells_any_event": 0.26782825559449985,
  "share_cells_event_in_half_of_weeks": 0.009975734699379887,
  "top1pct_cells_share_of_event_weeks": 0.3337666755765072,
  "weekly_share_min": 0.002580768495507551,
  "weekly_share_max": 0.04368509483997652
 }
}```
- **Panel (c) has sharp one-week dips.** Both series fall to about 0.3-0.6% in a handful of single weeks:
  around each new year (late Dec / early Jan), mid-2019, mid-2020, mid-2023, mid-2024 and Nov 2024. The last week
  (Dec 2024) is also low. That pattern looks like partial or split weeks, for example where a `week_start` straddles a
  year boundary in the original weekly aggregation, rather than real drops in activity. You may want to check that
  before the paper text describes the series.

## 2026-11-06 13:16 EST: Task 6 done (job 1693753, 2 tasks, rc 0, no Tracebacks)
`results_real/compare/compare_{conflict,unrest}.json` is committed. It compares the models like for like on the
same Africa + ME test rows (AUC-PR / AUC-ROC / log score):

| model | conflict | unrest |
|---|---|---|
| xgb (corrected) | 0.514 / 0.951 / 0.0614 | 0.536 / 0.950 / 0.0658 |
| xgb + target encoding | **0.534 / 0.959 / 0.0585** | **0.550 / 0.960 / 0.0627** |
| xgb cross-fit (country folds) | 0.444 / 0.916 / 0.0716 | 0.507 / 0.939 / 0.0697 |
| hybrid, iid cell | 0.521 / 0.958 / 0.0609 | 0.547 / 0.959 / 0.0634 |
| hybrid, barrier field | 0.521 / 0.959 / 0.0605 | 0.547 / 0.959 / 0.0634 |
| persistence | 0.446 | 0.451 |

On these rows, the target-encoded XGBoost has the best scores. The hybrid field and iid models are within about
0.01-0.02 AUC-PR of it, and both beat plain corrected XGB.

Still running: 1693751 `cs_global` (Task 7) and 1693752 `cs_glmm` (Task 8).

## 2026-11-06 13:20 EST: Task 2f rerun with the 50m borders (job 1693881, rc 0)
- I pulled the new command and reran with `paper/data/ne_50m_countries.geojson`.
- `paper/figures/data_events.png` and `.json` are committed. The JSON is identical to the 13:14 entry: the aggregates
  don't depend on the borders file.
- Visually checked: the borders and land background render correctly, and Greenland and Antarctica are drawn as
  uncovered land.
- The single-week dips in panel (c), noted in the 13:14 entry, are still there.

## 2026-11-06 13:45 EST: Task 2f, Africa + ME version, and the low-week check (job 1693963, rc 0)
- **Figure.** I ran `paper/make_data_figure.py` from ac8beb3. `paper/figures/data_events.png` and `.json` are
  committed and replace the global version. Visually checked.
- **Check.** The check script is mine; its output is committed as `results_real/data/weekcheck.json`. It contains
  cells per week, the first and last `week_start` of each of the 50 partitions, and the duplicate count. All counts are
  aggregate.

**(iii) Duplicates.** `(grid_id, week_start)` duplicated rows in clean_v2.parquet: **0**.

**(i) Rows per week are complete.** Every flagged week has exactly the same number of cells as its two neighbours on
each side:
- 41,848 cells per week (13,295 in the box) up to 2019;
- 49,399 in 2020;
- 59,344 (13,719 in the box) from late 2020 on.

So the dips are **not missing rows**. The event indicators are lower in those weeks.

**(ii) The dips sit on partition boundaries.** The 50 files hold 7-9 weeks each, and no week appears in two files.
- Conflict: **17 of the 19** flagged weeks are the first or last `week_start` of a file. If boundary weeks were
  flagged at random, you'd expect about 27% (2 boundary weeks out of about 7.3 per file).
- Examples of Conflict share in the box against a median of 0.0233:

| week | Conflict share | file boundary |
|---|---|---|
| 2017-12-26 | 0.0073 | first week of file 01 |
| 2018-12-18 | 0.0053 | first of 08 |
| 2019-07-16 | 0.0056 | last of 11 |
| 2024-06-04 | 0.0075 | first of 47 |
| 2019-02-12 | 0.0092 | last of 08 |

- The two exceptions are 2024-12-10 and 2024-12-17. They are the last weeks of the data, so presumably ACLED's
  coverage was incomplete at extraction time.
- Unrest: 59 weeks flagged. Most are 2018 weeks that sit below the median because unrest levels in 2018 were lower
  overall, not because of single dips. The deep single-week dips are again boundary weeks: 2017-12-26 0.0028,
  2018-05-29 0.0053, 2018-10-30 0.0059, 2018-12-18 0.0035, 2019-07-16 0.0036, 2020-03-24 0.0058, 2024-06-04 0.0071.

**Conclusion.** This supports your hypothesis. Weeks at a partition boundary keep all their rows, but the event
counts look partly counted. The likely cause is that each partition aggregated events within its own date range,
so a 7-day week that straddles the cut gets only the days inside the file.

## 2026-11-06 13:48 EST: Task 7, conflict done (job 1693751.1, no Tracebacks; maxvmem 23.1 GB); unrest (.2) still running
`results_real/spatial_global/conflict/{summary.json,variogram.csv,field.csv}` is committed.

| global, conflict | AUC-PR | AUC-ROC | log score | hyperparameters |
|---|---|---|---|---|
| XGB cross-fit (country folds) | 0.496 | 0.937 | 0.0466 | |
| + barrier field | 0.557 | 0.971 | 0.0399 | country sd 0.82, range 298 km, field sd 1.30 (fit time 284 s) |
| + iid cell | 0.557 | 0.971 | 0.0400 | country sd 1.40, cell sd 1.12 |

- Size: 15,303,100 rows (4,787,224 test), 46,031 cells and 16,962 barrier cells.
- Persistence AUC-PR: 0.481.
- Residual diagnostics: Moran's I 0.581. Within-country vs across-border correlation is 0.608 vs 0.440 (83,886 and 4,359 pairs).

## 2026-11-06 13:52 EST: fitted-vs-observed figure (job 1693985, rc 0)
- I ran `paper/make_fitted_figure.py` from 288a737 on the Task 4r `test_pred.parquet` files (Africa + ME,
  computed on clean.parquet from 2r). `paper/figures/real_fitted.png` is committed, after a visual check.
- In the calibration panels (c) and (f), both models track the diagonal above about 1%. Below about 0.5%:
  - the **hybrid model over-predicts**: for example, a bin mean near 1e-3 has observed frequency about 2e-4 (conflict)
    or about 4e-4 (unrest);
  - **cross-fitted XGBoost under-predicts** in the same range.
- You may want to mention that in the text. It could come from the country effect or the field lifting cells with
  near-zero risk.

## 2026-11-06 13:49 EST: Round 3 submitted (repo at 65bacfe)
- Deleted the Round 2 jobs 1693751.2 `cs_global` (unrest) and 1693752.1-2 `cs_glmm`. Their partial outputs in
  `$WORK/spatial_global/unrest` and `$WORK/glmm` will be overwritten.
- Jobs:
  - loader **1694066** (`ONLY_CLEAN=1`, writes `$WORK/clean_r3.parquet`);
  - held on the loader: xgb **1694067** (array 1-8), spatial **1694068**, global **1694070**, glmm **1694071**;
  - compare **1694069**, held on xgb and spatial;
  - check_duplicates **1694072** (independent);
  - sims **1694073** (`-t 10-61`), then merge **1694074** held on it;
  - post-loader checks **1694075**: the partition-boundary week check and `check_missing.py` on clean_r3.
- Split weeks: the raw-partition check (13:45 entry, `weekcheck.json`) found **no `week_start` in two files**. So
  there are no split weeks in the duplicate sense, and the loader change doesn't affect that. For that reason I didn't
  hold the downstream jobs behind the check. The partly counted boundary weeks (event dips) are a property of
  the source partitions and are still present. Job 1694075 repeats the check on clean_r3 (cells per week, duplicates),
  and I'll report it here.

## 2026-11-06 13:55 EST: Round 3 check_duplicates done (job 1694072, rc 0, about 3 min, no Traceback)
`results_real/data/check_duplicates.json` is updated and now also reports per distinct cell-week.
The row-level numbers are unchanged from Round 2. The new per-cell-week fields:

| | Conflict | Unrest |
|---|---|---|
| distinct cell-weeks | 34,292,260 | 34,292,260 |
| event cell-weeks with a leak in **any** copy | 0.0838 | 0.0513 |
| event cell-weeks with a leak in the **first** copy | **0.000** | **0.000** |
| AUC-PR of the original count, first copy | 0.442 | 0.438 |
| AUC-PR of the true count, first copy | 0.477 | 0.459 |

So keeping the first row of each cell-week, as the loader now does, removes the same-week leakage entirely.

## 2026-11-06 13:58 EST: Round 3 loader done (job 1694066, rc 0, 13:49-13:56 about 8 min, maxvmem 21.7 GB, no Traceback)
`results_real/data/clean_r3_report.json` is committed.

| | clean_v2 | clean_r3 |
|---|---|---|
| rows | 19,365,684 | **19,364,639** (-1,045) |
| cells | 59,344 | **59,339** (-5) |
| weeks | 366 | 366 |
| cell-weeks dropped before coverage | 7,078,822 | 7,078,822 |

- `share_rows_nat_from_first_copy`: **0.145**.
- `share_rows_gdp_pc_missing`: **0.282**. Please check this. It's much higher than the 0.014 missing share of
  `gdp` in clean_v2, so `gdp_pc` (World Bank GDP / summed GHSL population) seems to be missing for whole
  countries or years where gdp itself was present. That could be a country-code mismatch with the World Bank file, or a
  zero or NaN population sum.
- `share_rows_nat_from_interior_by_year`: 0.729 in every year from 2018 to 2024 (0 in 2017, the one December-2017 week).
- `countries_without_any_interior_cell`: [].
- Xgb 1694067 (8), spatial 1694068, global 1694070, glmm 1694071 and the post-load checks 1694075 are now running.

## 2026-11-06 14:01 EST: Round 3 post-load checks done (job 1694075, rc 0, about 3 min)
`results_real/data/missing_r3.json` and `results_real/data/weekcheck_r3.json` are committed.

**Week check on clean_r3.** There are **0** duplicated `(grid_id, week_start)` rows. Every flagged low week has the same
cell count as its neighbours: 41,848 up to 2019, 49,399 in 2020 and 59,339 afterwards. The partition files
are the same raw data, so the boundary-week event dips reported at 13:45 are still there. No `week_start` occurs in
two files.

**Missingness in the final clean_r3 table.**

| | all (19,364,639) | Africa + ME (4,976,634) |
|---|---|---|
| EPI_index / v2x_libdem | 0.0035 / 0.0038 | 0.0027 / 0.0027 |
| GHG / MF / ECO | **0.0058** (clean_v2: 0.178) | **0.0028** (clean_v2: 0.279) |
| gdp / gdp_pc | 0.014 / **0.014** | 0.036 / **0.036** (SSD, YEM, SOL, ERI, ESH, SYR, KOS) |
| electricity_access | 0.0038 | 0.0054 |
| IPC phase | 0.896 | 0.627 |
| pm25 | 0.556 | 0.555 |

- **Correction to my 13:58 note.** In the final table, `gdp_pc` is missing exactly where `gdp` is (1.4%), not 28%. The
  report's `share_rows_gdp_pc_missing = 0.282` must be computed on a different row set: before the coverage and land
  filters, or before the first-copy fallback. The report field is misleading but the data are fine.
- The new national-indicator fallback works: the sustainability trio dropped from 28% missing to 0.3% in the box.

## 2026-11-06 14:03 EST: Round 3 xgb done (job 1694067, 8 tasks on gpu, rc 0, no Tracebacks; maxvmem 68.6 GB)
- Tasks started at 13:57. The corrected and dedup tasks finished at 13:58, the original ones at 14:00-14:01.
- The 8 `metrics_*.json` files are in `results_real/xgb/`. Rounds and AUC-PR by group:

| run | rounds | AUC-PR | border | interior | onset AUC-PR | continuation AUC-PR | persistence | 2023 / 2024 AUC-PR |
|---|---|---|---|---|---|---|---|---|
| original Conflict | 1998 | 0.934 | 0.966 | 0.498 | None | None | 0.882 | 0.934 / 0.935 |
| original Unrest | 1474 | 0.917 | 0.968 | 0.528 | None | None | 0.843 | 0.917 / 0.918 |
| dedup Conflict | 488 | 0.546 | 0.590 | 0.523 | 0.008 | 0.581 | 0.481 | 0.544 / 0.548 |
| dedup Unrest | 646 | 0.555 | 0.543 | 0.558 | 0.009 | 0.580 | 0.468 | 0.557 / 0.554 |
| corrected Conflict | 637 | 0.544 | 0.588 | 0.521 | 0.007 | 0.580 | 0.481 | 0.543 / 0.545 |
| corrected Unrest | 814 | 0.557 | 0.545 | 0.559 | 0.010 | 0.581 | 0.468 | 0.559 / 0.554 |
| corrected_te Conflict | 381 | 0.566 | 0.598 | 0.549 | 0.011 | 0.596 | 0.481 | 0.569 / 0.563 |
| corrected_te Unrest | 135 | 0.583 | 0.567 | 0.586 | 0.012 | 0.605 | 0.468 | 0.588 / 0.577 |

- **original, scored on one row per cell-week** (4,787,224 test rows, the repaired test set): AUC-PR **0.473** for
  Conflict (persistence 0.450) and **0.479** for Unrest (persistence 0.443). That is down from 0.93 / 0.92 on the
  duplicated rows.
- In both original files, `one_row_per_cell_week` and `one_row_per_cell_week_repaired_test_set` are identical. Both
  have 4,787,224 rows and the same scores. If they are meant to be different row sets, check that the second one isn't
  silently falling back to the first.
- The onset AUC-PR values are about 0.007-0.012, and the onset rows key holds a single AUC-PR. Onsets are very rare,
  so compare these against onset prevalence, not against the overall numbers.

## 2026-11-06 14:15 EST: Round 3 spatial done (job 1694068, 2 tasks, rc 0, about 15 min; maxvmem 15.7 GB, no Tracebacks)
- Files in `results_real/spatial/{conflict,unrest}/`: `summary.json`, `variogram.csv` (test period), `variogram_train.csv`
  and `field.csv`.
- Model inputs: 4,976,634 rows and 13,719 cells. The offset is trained on the Africa + ME rows only.

| outcome | model | AUC-PR | AUC-ROC | log score | hyperparameters |
|---|---|---|---|---|---|
| conflict | XGB cross-fit | 0.461 | | | |
| conflict | + field | 0.519 | 0.959 | 0.0606 | country sd 0.76, range 263 km, field sd 1.48 |
| conflict | + iid | 0.519 | 0.959 | 0.0609 | country sd 1.37, cell sd 1.30 |
| unrest | XGB cross-fit | 0.493 | | | |
| unrest | + field | 0.540 | 0.958 | 0.0639 | country sd 1.01, range 65 km, field sd 1.25 |
| unrest | + iid | 0.540 | 0.958 | 0.0640 | country sd 1.06, cell sd 1.15 |

- Persistence AUC-PR: 0.446 for conflict, 0.451 for unrest.
- Cells with P(w > 0) above 0.95 / below 0.05: 0.189 / 0.030 for conflict, 0.116 / 0.026 for unrest.
- Moran's I, test vs training period: 0.539 vs 0.636 for conflict, 0.258 vs 0.197 for unrest.
- Test-period border contrast:

| outcome | within-country | across border (all) | across border, same fold | across border, different fold |
|---|---|---|---|---|
| conflict | 0.585 | 0.285 | 0.247 | 0.284 |
| unrest | 0.287 | 0.097 | 0.137 | 0.091 |

  The across-border pairs are 264 same-fold pairs out of 2,040.
- Compare (1694069) started at 14:13.

## 2026-11-06 14:17 EST: Round 3 compare done (job 1694069, 2 tasks, rc 0, about 1 min, no Tracebacks)
- Files in `results_real/compare/`: `compare_{conflict,unrest}.json` and `compare_{conflict,unrest}_variogram.csv`.
- Setup: same Africa + ME test rows for every model, 13,719 cells, 93 countries, reference model = xgb.
- Δlog score is model minus xgb, so negative is better. SEs are clustered by cell and by country. Moran's I is on the
  test-period residuals.

| model | conflict AUC-PR | Δlog (SE cell / SE country) | Moran's I | unrest AUC-PR | Δlog (SE cell / SE country) | Moran's I |
|---|---|---|---|---|---|---|
| xgb (corrected) | 0.515 | (ref) | 0.424 | 0.532 | (ref) | 0.148 |
| xgb_dedup | 0.518 | -0.00061 (0.00008 / 0.00053) | 0.354 | 0.534 | +0.000001 (0.00006 / 0.00015) | 0.156 |
| xgb_te | **0.533** | **-0.00350** (0.00023 / 0.00127) | **0.295** | **0.550** | **-0.00322** (0.00020 / 0.00078) | 0.253 |
| xgb_crossfit | 0.461 | +0.00824 (0.00042 / 0.00227) | 0.539 | 0.493 | +0.00474 (0.00025 / 0.00120) | 0.258 |
| hybrid_iid | 0.519 | -0.00093 (0.00020 / 0.00089) | 0.450 | 0.540 | -0.00204 (0.00017 / 0.00067) | 0.292 |
| hybrid_field | 0.519 | -0.00117 (0.00021 / 0.00094) | 0.449 | 0.540 | -0.00207 (0.00017 / 0.00066) | 0.287 |
| persistence | 0.446 | | | 0.451 | | |

- With country-clustered SEs, xgb_te beats xgb clearly for both outcomes (about 2.8 and 4.1 SE).
- The hybrid models beat xgb for unrest (about 3 SE). For conflict the gain is within about 1.3 SE.
- For unrest, the hybrid residuals have a higher Moran's I than the xgb residuals (0.29 vs 0.15).

## 2026-11-06 14:25 EST: Round 3 simulations and figures done
- **Sims** 1694073 (`-t 10-61`, 52 tasks, 13:41-14:10, maxvmem about 15.9 GB, no Tracebacks). Merge 1694074 had no FAILED scripts.
  - Committed the updated `results/s2_*.csv`, `s3_*.csv`, `s4_*.csv` and `results/s1_maps.png`.
  - New files: `results/s1_paired.csv`, `s1_paired_calibrated.csv`, `s1_raw_calibrated.csv`, `s1_scores_calibrated.csv`,
    `s4_extra_asym.csv`, `s4_raw_asym.csv` and `s4_summary_asym.csv`.
  - s0 and s1-base were not rerun, as asked: tasks 1-9 were skipped.
- **Figures and tables** (job 1694170, all rc 0):
  - `make_data_figure.py` on clean_r3: only `data_events.json` changed; the PNG is byte-identical.
  - `make_fitted_figure.py`: real_fitted.png.
  - `make_real_figures.py`: real_field.png, real_variogram.png and real_{xgb,diag,hybrid,field,boxes}.tex. I visually
    checked real_field.png.
  - `make_tables.py` and `make_sim_figures.py`: s2_estimates.png and s4_estimates.png. The s*.tex tables were rewritten
    but came out unchanged.
  - The real_* outputs use the Round 3 xgb, spatial and compare results. Global (1694070) and glmm (1694071) are
    still running. If make_real_figures should include them, tell me and I'll rerun it once they finish.

## 2026-11-06 15:21 EST: Round 3 global done (job 1694070, 2 tasks, rc 0, no Tracebacks)
- Run times: conflict 13:57-14:45 (about 48 min), unrest 13:57-15:19 (about 82 min). maxvmem about 25.4 GB.
- Files in `results_real/spatial_global/{conflict,unrest}/`: `summary.json`, `variogram.csv`, `variogram_train.csv` and
  `field.csv`.
- Model inputs: 15,333,584 rows (4,787,224 test), 46,031 cells and 16,962 barrier cells.

| outcome | XGB cross-fit | + field (AUC-PR / ROC / log) | + iid | persistence | field hyperparameters |
|---|---|---|---|---|---|
| conflict | 0.468 | **0.533** / 0.969 / 0.0409 | 0.533 / 0.969 / 0.0410 | 0.481 | country sd 0.75, range 313 km, sd 1.27 (813 s) |
| unrest | 0.508 | **0.573** / 0.964 / 0.0611 | 0.573 / 0.964 / 0.0612 | 0.468 | country sd 1.07, range 65 km, sd 1.14 (625 s) |

- Cells with P(w > 0) above 0.95 / below 0.05: 0.141 / 0.036 for conflict, 0.110 / 0.025 for unrest.
- Moran's I, test vs train: 0.593 vs 0.554 for conflict, 0.370 vs 0.310 for unrest.
- Test-period border contrast:

| outcome | within-country | across border (all) | same fold | different fold |
|---|---|---|---|---|
| conflict | 0.608 | 0.564 | **0.750** | 0.281 |
| unrest | 0.407 | 0.187 | 0.284 | 0.159 |

- **Please look at this.** For conflict, globally, the across-border correlation is 0.75 between cells in the same
  cross-fitting fold but only 0.28 across folds. The overall across-border figure, 0.564, is close to the within-country
  0.608, so it looks like the fold assignment, not the border, drives much of the residual correlation. In Africa + ME
  the split was flat: 0.247 vs 0.284.
- Glmm (1694071) is still running; both tasks are at maxvmem about 29.7 GB.

## 2026-11-06 16:06 EST: Round 4 submitted (repo at 5c5f542)
- 11_dynamic: **1694603** (array 1-8). 12_dynamic_compare: **1694604** (array 1-2), held on it.
- I'll report the wall time and maxvmem of each task, and flag any space-time task that looks set to exceed 24 h.
- Round 3 glmm (1694071) is still running.

## 2026-11-06 16:13 EST: Round 4 oracle done (1694603.1-2, rc 0, about 3 min each, maxvmem about 21 GB)
- Files in `results_real/dynamic/<outcome>/`: `summary_oracle.json`, `variogram_{offset,iid,field}.csv` and `field_test.csv`.
  These are from the oracle run. `field_test.csv` will be overwritten if later tasks write the same name, so I'll
  recopy it at the end.
- In-sample fit on the test period (2023-24), Africa + ME, 4,976,634 rows and 13,719 cells:

| outcome | offset alone | + iid (AUC-PR / log) | + field (AUC-PR / log) | Δlog-ML field − iid | field range | Moran's I: offset / iid / field |
|---|---|---|---|---|---|---|
| conflict | 0.461 / 0.0700 | 0.555 / 0.0570 | 0.554 / 0.0567 | **+1045** | 374 km | 0.539 / 0.346 / **-0.119** |
| unrest | 0.493 / 0.0707 | 0.579 / 0.0605 | 0.578 / 0.0604 | **+69** | 66 km | 0.258 / 0.154 / **0.003** (p = 0.33) |

- Even fitted on the test period itself, field and iid give the same scores. The marginal likelihood still strongly
  prefers the field: +1045 for conflict, +69 for unrest.
- The field removes the residual autocorrelation. For conflict it slightly over-corrects, to a negative Moran's I.
- Rolling (tasks 3-4) and space-time (tasks 5-8) have been running since 16:06. Glmm 1694071 is still running, at maxvmem about 33.6 GB.

## 2026-11-06 16:24 EST: Round 4 rolling done (1694603.3-4, rc 0; conflict about 16 min, unrest about 13 min; maxvmem about 21 GB)
`results_real/dynamic/<outcome>/summary_rolling.json` is committed. Training years are 2018-2023 and the test year is 2024.

| outcome | xgb cross-fit, rolling | + field (AUC-PR / log) | + iid (AUC-PR / log) | field range |
|---|---|---|---|---|
| conflict | 0.487 / 0.0682 | 0.535 / 0.0597 | 0.535 / 0.0600 | 248 km |
| unrest | 0.526 / 0.0695 | 0.572 / 0.0626 | 0.572 / 0.0626 | 66 km |

- The space-time tasks 5-8 have been running since 16:06 (about 16 min so far). They print no progress. I'll check
  their elapsed time and maxvmem every ~2 h and flag here if any looks set to exceed 24 h.

## 2026-11-06 16:28 EST: Round 4 space-time, unrest at 26-week steps done (1694603.8, rc 0, about 20 min wall; fit 1077 s; maxvmem about 21 GB)
`results_real/dynamic/unrest/summary_st26_r1.json` is committed.
- Setup: 3,637 one-degree cells, 15 steps (11 train), latent size 74,065.
- Scores: AUC-PR **0.540**, AUC-ROC 0.959, log score 0.0637. The static field in Round 3 spatial had 0.540 / 0.0639.
- Hyperparameters: country sd 0.94, cell sd 1.06, space-time range **891 km**, sd 0.63, rho **0.94**.
- **Run time is not a concern.** This task needed about 18 min. The 13-week tasks have twice as many steps, so I
  expect them to finish within a few hours, well under 24 h. They have been running for about 22 min so far.

## 2026-11-06 16:32 EST: Round 4 space-time, conflict at 26-week steps done (1694603.7, rc 0, about 24 min wall; fit 1297 s)
`results_real/dynamic/conflict/summary_st26_r1.json` is committed. Same setup as unrest: 3,637 cells, 15 steps, latent size 74,065.
- Scores: AUC-PR **0.518**, AUC-ROC 0.960, log score 0.0603, log-ML -183,089. The static field in Round 3 had
  AUC-PR 0.519 and log score 0.0606; its log-ML was -184,943, about 1,850 lower.
- Hyperparameters: country sd 0.79, cell sd 0.80, space-time range 731 km, sd 1.33, rho **0.974**.
- The 13-week tasks (5-6) have now run for about 25 min.

## 2026-11-06 18:03 EST: progress check
- Round 4 space-time at 13-week steps (1694603.5-6): wall time 1 h 55 min so far, maxvmem 21.2 GB, no Traceback. The
  26-week tasks needed about 20 min. With twice the steps and a latent size about 2x, a 2-hour run is plausible but
  slower than linear. I'll keep checking and flag here if either passes about 12 h, the halfway point to your 24 h limit.
- Round 3 glmm (1694071.1-2): wall time 4 h 05 min, maxvmem about 38.5 GB, no Traceback.

## 2026-11-06 18:23 EST: Round 4 space-time, conflict at 13-week steps done (1694603.5, rc 0, wall about 2 h 15 min; fit 7970 s; maxvmem 21.2 GB)
`results_real/dynamic/conflict/summary_st13_r1.json` is committed.
- Setup: 3,637 cells, 29 steps (21 train), latent size 124,983.
- Scores: AUC-PR **0.519**, AUC-ROC 0.959, log score 0.0602, log-ML -182,981.
  - Against the 26-week fit: 0.518 / 0.0603 / -183,089.
  - Against the static field: 0.519 / 0.0606 / -184,943.
- Hyperparameters: country sd 0.79, cell sd 0.81, space-time range 779 km, sd 1.39, rho **0.986**.
- The unrest 13-week task (1694603.6) is still running. It should be similar, about 2.5 h, so done around 18:40.

## 2026-11-06 18:29 EST: Round 4 space-time, unrest at 13-week steps done (1694603.6, rc 0, wall about 2 h 21 min; fit 8333 s; maxvmem 21.2 GB). All of 11_dynamic is done.
`results_real/dynamic/unrest/summary_st13_r1.json` is committed. Latent size 124,983 over 29 steps.
- Scores: AUC-PR **0.541**, AUC-ROC 0.959, log score 0.0637, log-ML -191,267.
  - Against the 26-week fit: 0.540 / 0.0637 / -191,541.
  - Against the static field: 0.540 / 0.0639 / -193,055.
- Hyperparameters: country sd 0.96, cell sd 1.07, space-time range 996 km, sd 0.62, rho 0.949.
- Wall-time summary for 11_dynamic:

| task | wall time |
|---|---|
| oracle | about 3 min |
| rolling | 13-16 min |
| space-time, 26-week steps | 20-24 min |
| space-time, 13-week steps | 2 h 15 min - 2 h 21 min |

  maxvmem was about 21 GB for every task. No task came close to 24 h.
- `field_test.csv` (from the oracle tasks) is unchanged since the 16:13 commit.
- 12_dynamic_compare (1694604.1-2) started at 18:27.

## 2026-11-06 18:31 EST: Round 4 dynamic compare done (1694604.1-2, rc 0, about 1 min, no Tracebacks). Round 4 is complete.
Files in `results_real/compare/`: `compare_{conflict,unrest}_{st,rolling}.json` and their `_variogram.csv` files.
Δlog is model minus xgb, so negative is better; the SE is clustered by country.

**Space-time comparison (2023-24 test rows, 1,426,776):**

| model | conflict AUC-PR / Δlog (SE) / Moran's I | unrest AUC-PR / Δlog (SE) / Moran's I |
|---|---|---|
| xgb_te | 0.533 / -0.00350 (0.00127) / 0.295 | 0.550 / -0.00322 (0.00078) / 0.253 |
| hybrid_iid | 0.519 / -0.00093 (0.00089) / 0.450 | 0.540 / -0.00204 (0.00067) / 0.292 |
| hybrid_field | 0.519 / -0.00117 (0.00094) / 0.449 | 0.540 / -0.00207 (0.00066) / 0.287 |
| hybrid_st13 | 0.519 / -0.00156 (0.00096) / 0.405 | 0.541 / -0.00233 (0.00079) / 0.288 |
| hybrid_st26 | 0.518 / -0.00152 (0.00095) / 0.412 | 0.540 / -0.00229 (0.00080) / 0.291 |

**Rolling comparison (713,388 rows, the 2024 rows only):**

| model | conflict AUC-PR / Δlog (SE) | unrest AUC-PR / Δlog (SE) |
|---|---|---|
| xgb (ref) | 0.522 | 0.553 |
| xgb_te | 0.533 / -0.00278 (0.00142) | 0.562 / -0.00234 (0.00096) |
| hybrid_field (static) | 0.520 / -0.00067 (0.00122) | 0.557 / -0.00173 (0.00067) |
| **hybrid_field_roll** | **0.535 / -0.00217 (0.00078)** | **0.572 / -0.00327 (0.00079)** |
| hybrid_iid_roll | 0.535 / -0.00185 (0.00076) | 0.572 / -0.00324 (0.00079) |
| xgb_crossfit_roll | 0.487 / +0.00627 (0.00146) | 0.526 / +0.00359 (0.00111) |

- Refitting through 2023 helps the hybrid on 2024 more than adding time dynamics does.
  - With the rolling refit, hybrid_field_roll matches or beats xgb_te on 2024 for unrest, and has the highest AUC-PR
    for conflict.
  - The space-time variants are within noise of the static field.
- **Label inconsistency.** In `compare_*_rolling.json`, `period` says "test (2023-2024)", but `n_rows` is 713,388:
  the 2024 rows only, half of 1,426,776. Please fix the label or the row set.
- Round 3 glmm (1694071) is still running: wall about 4 h 30 min, maxvmem 38.5 GB.

## 2026-11-06 19:11 EST: Round 5 submitted (repo at 5f72419)
- Jobs: xgb tasks 7-8 **1695628**; 13_update **1695629** (array 1-2); 14_update_compare **1695630** (array 1-2), held on both.
- Round 3 glmm 1694071 is still running.

## 2026-11-06 19:14 EST: Round 5 xgb tasks 7-8 done (1695628, rc 0, about 2 min, no Tracebacks)
`results_real/xgb/metrics_corrected_te_{Conflict,Unrest}.json` is committed. The main metrics are unchanged:
AUC-PR 0.566 / 0.583, with 381 and 135 rounds.

New `quarterly_update` block, on the global test set (4,787,224 rows):

| outcome | static te | quarterly-updated te (`logit_upd`) | persistence |
|---|---|---|---|
| Conflict | 0.566 | **0.578** (ROC 0.975) | 0.481 |
| Unrest | 0.583 | **0.588** (ROC 0.968) | 0.468 |

13_update (1695629.1-2) is running, at maxvmem 16.5 GB so far.

## 2026-11-06 19:46 EST: Round 6 submitted (repo at 657ddc5)
- Jobs: 15_s5 **1696020** (array); merge **1696021**, held on it.
- For the merge I used a small wrapper script with the same resources: `source crc/jobs/env.sh && cd sims && N_JOBS=1
  python s5_bivariate.py`. The `qsub -b y "source ... && ..."` form in TASKS.md would not go through a shell, so
  `source` and `&&` would fail.

## 2026-11-06 20:02 EST: Round 7 submitted (repo at 8d506cb)
- Job: 16_bivariate **1696206** (tasks 1-2).
- It requests mem_free=200G. Most long-queue nodes have 251 GB, so it may wait for a mostly idle node. I'll report
  the queue wait if it's long, and task 2's maxvmem after its first hour.

## 2026-11-06 20:04 EST: Round 6 S5 done (1696020, 20 tasks, about 10-15 min each, maxvmem about 2.5 GB, no Tracebacks; merge 1696021 rc 0)
`results/s5_raw.csv` and `results/s5_summary.csv` are committed. 20 replicates per scenario; the shard checkpoints are not committed.
```
     scenario     model  truth  mean_est  emp_sd  mean_se  coverage95  reject_zero   d_c  log_score  auc_pr  onset_auc_pr  loading  seconds  n  mc_se_coverage
   escalation    biv_st    0.4     0.434   0.111    0.126        0.95         0.95 0.358      0.140   0.103         0.042    0.832  316.759 20           0.049
   escalation       glm    0.4     1.026   0.163    0.116        0.00         1.00 1.516      0.143   0.077         0.025      NaN    0.026 20           0.000
   escalation uni_field    0.4     0.794   0.128    0.122        0.10         1.00 0.805      0.139   0.105         0.044      NaN    0.419 20           0.067
   escalation    uni_st    0.4     0.659   0.108    0.127        0.50         1.00 0.375      0.140   0.105         0.045      NaN   80.759 20           0.112
no_escalation    biv_st    0.0     0.042   0.166    0.138        0.90         0.10 0.376      0.134   0.088         0.033    0.827  309.529 20           0.067
no_escalation       glm    0.0     0.695   0.193    0.129        0.05         0.95 1.539      0.137   0.063         0.022      NaN    0.033 20           0.049
no_escalation uni_field    0.0     0.426   0.161    0.135        0.15         0.85 0.821      0.133   0.090         0.037      NaN    0.409 20           0.080
no_escalation    uni_st    0.0     0.269   0.158    0.140        0.45         0.55 0.384      0.134   0.090         0.034      NaN   79.353 20           0.111
```
- Only biv_st recovers the escalation coefficient:
  - Truth 0.4: estimate 0.434, coverage 0.95.
  - Truth 0: estimate 0.042, coverage 0.90, false rejection 0.10.
- The univariate models and glm are biased upward and falsely reject zero 55-95% of the time when there is no
  escalation. The ranking is glm > uni_field > uni_st.
- Predictive scores are close across the latent models.

## 2026-11-06 21:04 EST: Round 7 bivariate, 1-hour memory check (1696206, both tasks on d32cepyc246, 251 GB node)
| task | region | wall time | maxvmem | progress (log) |
|---|---|---|---|---|
| 1 | west_sahel: 906,810 rows per outcome, 2,505 cells, v on 641 cells × 28 steps | 1 h 00 min | **22.6 GB** | uni_static and uni_st done |
| 2 | africa_me: 4,921,758 rows per outcome, 13,719 cells, v on 3,637 cells × 14 steps | 1 h 00 min | **43.2 GB** (it jumped from about 31 GB at 55 min) | uni_static done |

- Node memory use is about 77 GB of 251 GB, so there is no memory pressure. Neither task has a Traceback.
- I can't estimate the total wall time for task 2 yet, because the log only shows completed models. I'll check again
  in about 2 h and flag here if it looks likely to exceed 48 h.

## 2026-11-06 22:02 EST: Round 5 update, conflict done (1695629.1, rc 0, 19:11-21:54 about 2 h 43 min, no Traceback; maxvmem up to 26.9 GB)
`results_real/dynamic/conflict/summary_update.json` is committed. Each model is updated quarterly over 8 quarters (2023 Q1 to 2024 Q4),
on the Africa + ME test rows:

| model | AUC-PR | log score | AUC-ROC |
|---|---|---|---|
| field (static hyperparameters, updated conditioning) | 0.531 | 0.0592 | 0.962 |
| iid | 0.531 | 0.0595 | 0.961 |
| **st (13-week steps)** | **0.539** | **0.0583** | **0.963** |

For comparison, the static fit of the same models (Round 3/4, no updating) gave 0.519 / 0.0606.

Still running: the unrest update (1695629.2). The update compare (1695630) is held on it.

## 2026-11-06 22:30 EST: Round 5 complete
**Unrest update** (1695629.2): rc 0, 19:11-22:26 about 3 h 15 min, maxvmem 26.9 GB.
`results_real/dynamic/unrest/summary_update.json` is committed. Field 0.550 / 0.0629, iid 0.550 / 0.0630, **st 0.554 / 0.0626** (AUC-PR / log score).

**Update compare** (1695630.1-2): rc 0, about 1 min, no Tracebacks. `results_real/compare/compare_{conflict,unrest}_update.json`
and the `_variogram.csv` files are committed. Test rows 2023-24: 1,426,776. Δlog is model minus xgb (country-clustered SE).

| model | conflict AUC-PR / log / Δlog (SE) | unrest AUC-PR / log / Δlog (SE) |
|---|---|---|
| xgb (ref) | 0.515 / 0.0618 | 0.532 / 0.0660 |
| xgb_te (static) | 0.533 / 0.0583 / -0.00350 (0.00127) | 0.550 / 0.0628 / -0.00322 (0.00078) |
| **xgb_te_upd** | **0.544 / 0.0568 / -0.00498 (0.00131)** | **0.559 / 0.0616 / -0.00438 (0.00083)** |
| hybrid_field (static) | 0.519 / 0.0606 / -0.00117 (0.00094) | 0.540 / 0.0639 / -0.00207 (0.00066) |
| hybrid_field_upd | 0.531 / 0.0592 / -0.00265 (0.00077) | 0.550 / 0.0629 / -0.00306 (0.00072) |
| hybrid_iid_upd | 0.531 / 0.0595 / -0.00230 (0.00076) | 0.550 / 0.0630 / -0.00303 (0.00072) |
| hybrid_st_upd | 0.539 / 0.0583 / -0.00352 (0.00081) | 0.554 / 0.0626 / -0.00339 (0.00071) |

- Quarterly updating improves every model by about 0.01 AUC-PR.
- Among the latent models, the space-time model now beats the static field. The updated location-aware XGBoost is
  still best overall.
- Still running: Round 7 bivariate (1696206) and Round 3 glmm (1694071).

## 2026-11-07 00:26 EST: Round 7 bivariate, progress check (1696206)
| task | wall time | maxvmem | models finished (log) |
|---|---|---|---|
| 1 west_sahel | 4 h 22 min | 22.6 GB | uni_static, uni_st (both done within the first hour) |
| 2 africa_me | 4 h 22 min | **59.2 GB** | uni_static, uni_st |

- No Tracebacks. The node (d32cepyc246, 251 GB) has plenty of headroom.
- Both tasks have been on the bivariate model(s) for more than 3 h. On west_sahel, the univariate space-time fit took
  under an hour, so the bivariate fit is much slower.
- The log only reports finished models, so I can't project task 2's total. If it hasn't finished the bivariate fit by
  about 12 h of wall time (around 08:00 EST), I'll flag that the 48 h limit is at risk.
- Round 3 glmm 1694071 is still running: about 10 h 20 min wall, maxvmem 39.9 GB.

## 2026-11-07 02:30 EST: Round 7 bivariate west_sahel done (1696206.1, rc 0, about 5 h 23 min wall, maxvmem 22.6 GB, no Traceback)
`results_real/bivariate/west_sahel/summary_biv.json` is committed. Data: 906,810 rows per outcome, 2,505 cells and 22 countries;
v on 641 cells x 28 steps.

| model | fit time | log-ML | conflict test AUC-PR / log | unrest test AUC-PR / log |
|---|---|---|---|---|
| uni_static | 493 s | -49,530 | 0.470 / 0.1020 | |
| uni_st | 2,606 s | -49,081 | 0.471 / 0.1026 | |
| biv | **16,253 s** | -76,592 (both outcomes) | 0.466 / 0.1031 | 0.289 / 0.0490 |

**Lag coefficients** (estimate, SE):

| term | uni_static | uni_st | biv |
|---|---|---|---|
| conflict:lag_own | 0.575 (0.016) | 0.444 (0.017) | 0.459 (0.017) |
| conflict:lag_other | 0.138 (0.031) | 0.131 (0.032) | **0.044 (0.032)** |
| conflict:lag_own_nb | 0.129 (0.007) | 0.051 (0.008) | 0.061 (0.008) |
| conflict:lag_other_nb | 0.005 (0.008) | 0.003 (0.008) | -0.020 (0.008) |
| unrest:lag_own | | | 0.351 (0.029) |
| unrest:lag_other | | | 0.066 (0.031) |
| unrest:lag_own_nb | | | 0.017 (0.009) |
| unrest:lag_other_nb | | | -0.042 (0.013) |

**Hyperparameters:**
- uni_static: country sd 0.688, cell sd 0.646, field range 514 km, field sd 1.698.
- uni_st: country 0.666, cell 0.638, field 508 km / 1.744; st range 423 km, sd 0.686, rho 0.905.
- biv:
  - country sd: conflict 0.691, unrest 0.528;
  - cell sd: conflict 0.661, unrest 1.335;
  - field_conflict: 453 km, sd 1.387;
  - field_unrest: 369 km, **sd 0.213**;
  - shared_st: range 499 km, sd 1.010, rho 0.952, **loading 0.868**.

The bivariate model's shared space-time component absorbs the unrest-to-conflict lag. lag_other drops from 0.138 to
0.044, within 1.4 SE of 0, which matches the S5 pattern. The predictive scores for conflict are about the same.

## 2026-11-07 02:30 EST: Round 3 glmm, unrest done (1694071.2, rc 0, 13:57-02:12 about 12 h 15 min; seconds_total 44,113; maxvmem 39.9 GB; no Traceback)
`results_real/glmm/{models_unrest.json, coefficients_unrest.csv, svc_tmax_unrest.csv}` is committed.

| model | test AUC-PR | log score | hyperparameters | fit time |
|---|---|---|---|---|
| glm | 0.518 | 0.0687 | | 147 s |
| country | 0.522 | 0.0676 | country 0.92 | 867 s |
| field | 0.535 | 0.0642 | country 0.92, range 465 km, sd 0.86, cell 1.28 | 6,427 s |
| svc | 0.534 | 0.0642 | as field, plus svc range 972 km, **svc sd 0.072** | 6,828 s |
| spatialplus m = 10 / 20 / 176 | 0.535 / 0.535 / 0.535 | 0.0642 | range 464 / 480 / 492 km | 6,183 / 7,569 / 6,561 s |

- **st part** (central_sahel: 249,744 rows, 968 cells, 33 steps of 8 weeks; it completed, with no `--st-step 16` rerun
  needed):
  - field: country 0.43, range 333 km, sd 0.30, cell 1.38;
  - field+st: st range 485 km, sd 0.51, **rho 0.49**.
- `prep.climate.*.cell_months_sd_zero`: tmax_z 0, tmin_z 0, ppt_z 2, **drought_z 138,600**, pm25_z 0.
- `prep.missing_indicators`:
  - kept: ipc_missing, log_gdp_pc_missing, electricity_access_missing, GHG_pc_t_missing;
  - EPI_index_missing was dropped because it is constant in training;
  - MF and ECO were merged into GHG_pc_t_missing because they are identical.
- **Please check `drought_z`.** 138,600 cell-months have SD 0, so the drought anomaly is degenerate in many cells.
  The floored SD presumably handles it, but it may be worth confirming.
- Still running: glmm conflict (1694071.1, about 12.5 h) and bivariate africa_me (1696206.2). That task has run
  about 6 h 20 min at 59 GB maxvmem. Given west_sahel's 4.5 h bivariate fit, africa_me (5.4x the rows) could take well
  over a day. I'll flag at 12 h of wall time (08:00 EST) if it hasn't finished.

## 2026-11-07 02:35 EST: Round 3 glmm, conflict done (1694071.1, rc 0, 13:57-02:27 about 12 h 30 min; seconds_total 45,009; maxvmem 39.9 GB; no Traceback)
`results_real/glmm/{models_conflict.json, coefficients_conflict.csv, svc_tmax_conflict.csv}` is committed.

| model | test AUC-PR | log score | hyperparameters | fit time |
|---|---|---|---|---|
| glm | 0.501 | 0.0665 | | 148 s |
| country | 0.511 | 0.0633 | country 1.19 | 917 s |
| field | 0.510 | 0.0607 | country 0.50, range 544 km, sd 1.72, cell 0.90 | 6,655 s |
| svc | 0.509 | 0.0608 | as field, plus svc range 465 km, **svc sd 0.076** | 6,154 s |
| spatialplus m = 10 / 20 / 138 | 0.510 / 0.510 / 0.510 | 0.0607 | range 545 / 537 / 575 km | 7,982 / 8,683 / 6,557 s |

- **st part** (central_sahel: 249,744 rows, 968 cells, 33 steps of 8 weeks; completed with no rerun needed):
  - field: country 0.20, range 430 km, sd 1.72, cell 0.53;
  - field+st: st range 230 km, sd 0.99, **rho 0.93**.
- `prep.climate.*.cell_months_sd_zero`: same as unrest. tmax 0, tmin 0, ppt 2, **drought 138,600**, pm25 0.
- `prep.missing_indicators`:
  - kept: ipc, log_gdp_pc, electricity_access, GHG_pc_t;
  - **EPI_index and v2x_libdem missing were dropped** because they are constant in training;
  - MF and ECO were merged into GHG.
- **Round 3 is now complete.** The only job still running is the Round 7 bivariate africa_me (1696206.2).

## 2026-11-07 08:21 EST: FLAG. Round 7 bivariate africa_me (1696206.2) may exceed 48 h
- Wall time 12 h 18 min, maxvmem 59.2 GB, no Traceback. uni_static and uni_st are done, and the bivariate fit has
  been running for roughly 8-9 h.
- Rough projection: the west_sahel bivariate fit took 16,253 s (4.5 h). africa_me has about 5.4x the rows,
  5.5x the static-field cells, and a space-time latent about 2.8x larger (3,637 x 14 against 641 x 28).
  - If cost scales 5-15x, the bivariate fit alone needs about 22-70 h.
  - So the total is plausibly 1-3 days, with a real chance of passing 48 h.
- `h_rt` is 96 h, so the job is killed at the latest around Mon 2026-11-10 20:00 EST.
- I am **leaving it running**. If you'd rather switch to sub-regions, put that in TASKS.md and say whether to qdel 1696206.2.

## 2026-11-07 09:51 EST: Round 7b submitted (repo at c7d9bd0)
- Deleted 1696206.2, the africa_me bivariate on the old engine, after about 13 h 40 min. It had not finished and
  wrote no output.
- Resubmitted as **1699951** (`-t 2`) with **`-pe smp 4`, not 12**. The CRC rule the user passed on is at most 4 cores
  per job, so I overrode the script's `-pe smp 12` on the command line. `FIT_JOBS=${NSLOTS}` = 4.
  Please change the header in `16_bivariate.sh` to 4 so the repo matches.
- scikit-sparse is not installed: it's optional, and installing into the env needs a separate batch job. Tell me if
  you want it, and I'll install it before any later rerun.
- I'll report the wall time.

## 2026-11-07 11:50 EST: Round 7b bivariate africa_me (1699951.2), first check
- Started 10:58 on d32cepyc105 (64 cores, 251 GB), with 4 slots.
- After 50 min: uni_static is done, which took under an hour as before.
- Memory: `maxvmem` 221.6 GB, `maxrss` 132.8 GB, current rss 130 GB.
  - vmem counts each of the 4 FIT_JOBS worker processes separately, so pages they share are counted 4 times.
    Physical use is the rss, about 130 GB.
  - qhost reports the node at only about 52 GB used, so there is no OOM risk right now.
  - The old engine peaked at 59 GB. If rss keeps growing past about 200 GB, I'll flag it.
    `FIT_JOBS` could then be capped below `NSLOTS`.

## 2026-11-07 12:00 EST: CRC core limits, noted
- Pulled f71ba46. 1699951.2 keeps running with 4 slots.
- For context, this is the crcfe01 login banner (/etc/motd):
  > "crcfe01, crcfe02, and bastion.crc.nd.edu are for interactive/development use only. Please limit the number of
  > cores used to 4 or less. ... IMPORTANT: USERS WHICH RUN JOBS USING MORE THAN 4 CORES - OR OVERLOAD FRONTEND
  > MACHINES MAY HAVE THEIR ACCOUNT SUSPENDED"

  It sits in the section about the front-end machines, so it most likely refers to interactive work there, not to
  batch jobs. That fits the documentation you quote, but the wording is ambiguous.
- The risk is account suspension, so I'm confirming the switch to `-pe smp 16` with Stefano directly before the next
  heavy submission. Nothing is waiting on this right now.

## 2026-11-07 12:15 EST: core limit confirmed by Stefano: up to 16 per job
- I reread docs.crc.nd.edu (crc_uge_env.html, quick_start.html). They agree with your summary:
  - long queue: 15 days, no limit on cores; `smp` up to 64 on one 64-core / 256 GB node;
  - at most 50 running jobs and 2,000 tasks per array;
  - one thread per core.
- The 4-core and one-hour limits apply to the front ends only ("Do not launch large mpi or smp processes on a front end").
- Two small differences:
  - I couldn't find the "1 core per 4 GB" rule on those pages. It does match the node ratio of 256 GB / 64 cores.
  - The scheduler here actually enforces 100 running jobs per user (`maxujobs`), not 50.
- From now on I'll use the job scripts as you write them, up to `-pe smp 16`, with FIT_JOBS × threads ≤ NSLOTS.
  1699951.2 keeps running on 4 slots.

## 2026-11-07 13:48 EST: Round 7b bivariate africa_me (1699951.2), progress
- Wall time 2 h 48 min. uni_static and uni_st are both done, faster than the old engine, which needed about 4 h for
  the same two models. The bivariate fit is now running. No Traceback.
- Memory as Grid Engine sums it over the 4 FIT_JOBS processes: rss 182 GB (maxrss 187 GB), vmem 316 GB.
  qhost reports the node at only 61.5 GB physically used, so the summed rss counts shared pages several times.
  There is no OOM sign. For future runs on 16 slots with FIT_JOBS=8, though, expect the summed figures to roughly
  double: rss about 370 GB on a 251 GB node. That would be fine only if the sharing holds.

## 2026-11-07 15:08 EST: Round 8 submitted (repo at 65d48ff)
- West Africa biv2: **1701825** (`-t 1`, 16 slots, FIT_JOBS=8), submitted first as the priority.
- S5 v2: **1701826** (40 tasks × 4 slots), with merge **1701827** held on it. The merge uses the same wrapper as Round 6:
  `cd sims && N_JOBS=1 python s5_bivariate.py`. The new checkpoints (`s5v2_ckpt*`) are separate from Round 6's `s5_ckpt*`.
- 1699951.2 (africa_me biv) keeps running. `17_biv2.sh -t 2` waits until it finishes.

## 2026-11-07 16:55 EST: Round 8 S5 v2 done (1701826, 40 tasks, about 1-1.7 h each, no Tracebacks; merge 1701827 rc 0)
`results/s5_raw.csv` and `results/s5_summary.csv` are committed: 4 scenarios × 5 models × 20 replicates.
```
         scenario     model  truth  mean_est  emp_sd  mean_se  coverage95  reject_zero   d_c  log_score  auc_pr  onset_auc_pr  loading  seconds  n  mc_se_coverage
       escalation    biv_st    0.4     0.434   0.111    0.126        0.95         0.95 0.358      0.140   0.103         0.042    0.832  187.833 20           0.049
       escalation   biv_st2    0.4     0.450   0.109    0.126        0.95         0.95 0.347      0.140   0.104         0.044    0.780 1411.987 20           0.049
       escalation       glm    0.4     1.026   0.163    0.116        0.00         1.00 1.516      0.143   0.077         0.025      NaN    0.096 20           0.000
       escalation uni_field    0.4     0.794   0.128    0.122        0.10         1.00 0.805      0.139   0.105         0.044      NaN    0.628 20           0.067
       escalation    uni_st    0.4     0.659   0.108    0.127        0.50         1.00 0.375      0.140   0.105         0.045      NaN   60.461 20           0.112
   escalation_uST    biv_st    0.4     0.385   0.125    0.118        0.90         0.90 0.444      0.141   0.105         0.043    1.172  182.167 20           0.067
   escalation_uST   biv_st2    0.4     0.467   0.137    0.120        0.85         1.00 0.350      0.141   0.106         0.045    0.806  998.065 20           0.080
   escalation_uST       glm    0.4     1.017   0.175    0.108        0.00         1.00 1.505      0.143   0.078         0.031      NaN    0.037 20           0.000
   escalation_uST uni_field    0.4     0.786   0.140    0.115        0.15         1.00 0.804      0.140   0.107         0.048      NaN    0.699 20           0.080
   escalation_uST    uni_st    0.4     0.679   0.136    0.122        0.35         1.00 0.372      0.140   0.107         0.046      NaN   65.750 20           0.107
    no_escalation    biv_st    0.0     0.042   0.166    0.138        0.90         0.10 0.376      0.134   0.088         0.033    0.827  287.443 20           0.067
    no_escalation   biv_st2    0.0     0.058   0.165    0.139        0.85         0.15 0.363      0.134   0.089         0.035    0.772 2010.460 20           0.080
    no_escalation       glm    0.0     0.695   0.193    0.129        0.05         0.95 1.539      0.137   0.063         0.022      NaN    0.216 20           0.049
    no_escalation uni_field    0.0     0.426   0.161    0.135        0.15         0.85 0.821      0.133   0.090         0.037      NaN    0.950 20           0.080
    no_escalation    uni_st    0.0     0.269   0.158    0.140        0.45         0.55 0.384      0.134   0.090         0.034      NaN   99.399 20           0.111
no_escalation_uST    biv_st    0.0    -0.012   0.166    0.131        0.85         0.15 0.467      0.134   0.088         0.038    1.191  278.204 20           0.080
no_escalation_uST   biv_st2    0.0     0.064   0.169    0.133        0.85         0.15 0.366      0.134   0.089         0.036    0.806 1486.684 20           0.080
no_escalation_uST       glm    0.0     0.683   0.201    0.121        0.10         0.90 1.526      0.137   0.063         0.025      NaN    0.056 20           0.067
no_escalation_uST uni_field    0.0     0.411   0.164    0.128        0.20         0.80 0.819      0.134   0.089         0.037      NaN    0.942 20           0.089
no_escalation_uST    uni_st    0.0     0.281   0.158    0.135        0.45         0.55 0.383      0.134   0.090         0.035      NaN  100.693 20           0.111
```
- When unrest has episodes of its own (`_uST` scenarios), **biv_st still recovers the escalation effect, with a small
  downward bias**: escalation_uST 0.385 against a truth of 0.4, coverage 0.90, power 0.90. Its loading rises to about
  1.17, which absorbs part of the unrest-specific signal.
- **biv_st2** (with an unrest-specific space-time field) is close to unbiased in the escalation scenarios (0.450 and
  0.467) but slightly over-covers. With no escalation, its false-rejection rate is 0.15, against 0.10-0.15 for biv_st.
- The univariate models and glm stay strongly biased upward in every scenario.
- biv_st2 costs about 5-7× biv_st in fit time: about 1,000-2,000 s per replicate against about 180-290 s.
- West Africa biv2 (1701825) is still running: about 1 h 45 min wall time, maxrss 65 GB, on an ACMS 24-core node.

## 2026-11-07 18:20 EST: Round 7b bivariate africa_me done (1699951.2, rc 0, 10:58-18:16, **7 h 18 min** on 4 slots with the new engine; maxrss 187 GB summed over workers; no Traceback)
`results_real/bivariate/africa_me/summary_biv.json` is committed. Data: 4,921,758 rows per outcome, 13,719 cells, 93 countries; v on 3,637 cells × 14 steps.

| model | fit time | log-ML | conflict test AUC-PR / log | unrest test AUC-PR / log |
|---|---|---|---|---|
| uni_static | 532 s | -173,069 | 0.524 / 0.0593 | |
| uni_st | 2,466 s | -170,579 | 0.522 / 0.0600 | |
| biv | **23,084 s** (6.4 h) | -354,669 (both outcomes) | 0.526 / 0.0595 | 0.541 / 0.0640 |

**Lag coefficients** (estimate, SE):

| term | uni_static | uni_st | biv |
|---|---|---|---|
| conflict:lag_own | 0.794 (0.008) | 0.603 (0.009) | 0.626 (0.009) |
| conflict:lag_other | 0.150 (0.014) | 0.133 (0.015) | **0.057 (0.015)** |
| conflict:lag_own_nb | 0.175 (0.004) | 0.070 (0.004) | 0.083 (0.004) |
| conflict:lag_other_nb | -0.004 (0.006) | -0.007 (0.006) | -0.035 (0.006) |
| unrest:lag_own | | | 0.585 (0.009) |
| unrest:lag_other | | | 0.025 (0.014) |
| unrest:lag_own_nb | | | 0.071 (0.004) |
| unrest:lag_other_nb | | | -0.042 (0.006) |

**Hyperparameters:**
- uni_static: country 0.497, cell 0.958, field 562 km / sd 1.63.
- uni_st: country 0.502, cell 0.971, field 583 km / 1.60; st 480 km, sd 0.869, rho 0.873.
- biv:
  - country sd: conflict 0.533, unrest 0.903;
  - cell sd: conflict 0.994, unrest 1.308;
  - field_conflict: 632 km, sd 1.197;
  - field_unrest: 439 km, sd 0.350;
  - shared_st: 486 km, sd 1.254, rho 0.946, **loading 0.668**.

As in west_sahel, the shared episode field absorbs most of unrest → conflict. lag_other falls from 0.150 to 0.057,
but unlike west_sahel it stays about 3.8 SE from zero.

**Next:** I submitted `17_biv2.sh -t 2` (africa_me biv2) as **1705023** now that 1699951.2 has finished, as TASKS.md says.
West Africa biv2 (1701825) has run 3 h 10 min, maxrss 65 GB.

## 2026-11-07 19:27 EST: Round 9 submitted (repo at f53144c)
- 18_joint_pred: **1706488** (tasks 1-2, 16 slots each).
- S5 v3: **1706489** (40 tasks × 4 slots; checkpoints `s5v3_ckpt*`). Merge **1706490** is held on it and uses the same wrapper as before.
- The biv2 jobs 1701825 and 1705023 keep running.

## 2026-11-07 19:35 EST: Round 9 joint prediction FAILED (1706488.2, africa_me), bug in fit_bivariate.py
```
  File "realdata/fit_bivariate.py", line 251, in main
    m = fit_or_condition(LatentGaussianModel(P["y"], uni_blocks(o, st), uni_designs(P, st)), saved, model)
  File "realdata/fit_bivariate.py", line 104, in fit_or_condition
    m.theta = theta_from_hyper(m.blocks, saved["models"][model]["hyper"])
  File "realdata/fit_bivariate.py", line 89, in theta_from_hyper
    th += [np.log(hyper[f"{nm}.range_km"]), np.log(hyper[f"{nm}.sd"]), np.arctanh(hyper[f"{nm}.rho"])]
KeyError: 'st_conflict.range_km'
```
- **Cause:** line 248 has `st = model.startswith("uni_st")`, which is **True for "uni_static"**. So uni_static is built with a
  space-time block and then looks for `st_conflict.*` in the saved uni_static hyperparameters, which don't have them.
  `summary_biv.json` itself is intact: uni_static has 4 hyper keys, and uni_st has the 3 `st_conflict.*` keys.
- **Suggested fix:** `st = model in ("uni_st", "uni_st_u")`. The same pattern would also misclassify "uni_static_u".
- Task 1 (west_sahel, 1706488.1) runs the same code, so it will very likely fail the same way. I'm leaving it running
  and will report when it ends. I won't resubmit until you push a fix.
- No problem with S5 v3 (1706489, 40 tasks running) or the field diffusion job.
- **biv2 africa_me (1705023):** rss summed over the 8 FIT_JOBS workers is 285 GB, which exceeds the 251 GB node. But qhost
  shows only 63 GB physically used on d12chas219, so this is shared memory counted once per worker; there is no OOM.
  Wall time is 1 h 11 min so far.
- 19:30 EST: task 1 (west_sahel, 1706488.1) failed the same way: `KeyError: 'st_conflict.range_km'`, same traceback.
  Neither region has a joint-prediction result. Once you push the fix, I'll pull and resubmit `18_joint_pred.sh` (both tasks).

## 2026-11-07 19:35 EST: Round 9 joint prediction resubmitted with the fix (2f8be0d)
- 18_joint_pred: **1706562** (tasks 1-2). S5 v3, the biv2 jobs and the merge are unchanged.

## 2026-11-07 19:40 EST: can't message you directly
I tried `send_message` to session_01H85kxLsrzzXKJMbCCVADs7 as you asked. It was refused again:
"elevated_session_untrusted_device: the session you named needs proof of a trusted device, and the server would not
vouch for this session". This CRC session runs on crcfe01, which isn't a trusted device, so it can't send to your
session. Messages from you to me do work.

So this file stays the channel. To pick up new entries without the PI relaying, you could poll
`git log origin/main -- crc/STATUS.md`, or read this session's transcript, as the geostatistics session does.
Every failure (with traceback) and every round summary will keep going here as soon as it happens.

## 2026-11-07 21:05 EST: Round 9 S5 v3 done (1706489, 40 tasks, no Tracebacks; merge 1706490 rc 0)
Committed `results/s5_raw.csv`, `s5_summary.csv` (unchanged estimates, as expected) and the new **`s5_prediction.csv`**:
```
         scenario   model  n  joint_log_score  joint_log_score_indep  cond_log_score  cond_auc_pr  both_auc_pr  both_auc_pr_indep  joint_log_score_minus_uni  joint_log_score_minus_uni_se  cond_log_sco
       escalation  biv_st 20           0.2890                 0.2892          0.1394       0.1038       0.0275             0.0273                    -0.0009                        0.0005              
       escalation biv_st2 20           0.2889                 0.2891          0.1392       0.1052       0.0275             0.0277                    -0.0010                        0.0005              
       escalation  uni_st 20           0.2899                    NaN          0.1399       0.1046       0.0320                NaN                     0.0000                        0.0000              
   escalation_uST  biv_st 20           0.3063                 0.3064          0.1399       0.1044       0.0321             0.0329                    -0.0013                        0.0004              
   escalation_uST biv_st2 20           0.3063                 0.3065          0.1397       0.1062       0.0325             0.0333                    -0.0012                        0.0004              
   escalation_uST  uni_st 20           0.3075                    NaN          0.1403       0.1070       0.0349                NaN                     0.0000                        0.0000              
    no_escalation  biv_st 20           0.2832                 0.2834          0.1335       0.0884       0.0242             0.0240                    -0.0010                        0.0005              
    no_escalation biv_st2 20           0.2831                 0.2833          0.1334       0.0897       0.0249             0.0243                    -0.0011                        0.0005              
    no_escalation  uni_st 20           0.2842                    NaN          0.1341       0.0898       0.0283                NaN                     0.0000                        0.0000              
no_escalation_uST  biv_st 20           0.3000                 0.3002          0.1336       0.0882       0.0273             0.0269                    -0.0014                        0.0005              
no_escalation_uST biv_st2 20           0.3002                 0.3004          0.1336       0.0892       0.0260             0.0266                    -0.0013                        0.0004              
no_escalation_uST  uni_st 20           0.3014                    NaN          0.1342       0.0896       0.0284                NaN                     0.0000                        0.0000              
rc=0
```
- **Joint prediction:** biv_st and biv_st2 improve the joint log score over the product of the univariate models
  (uni_st) by about 0.001 (SE 0.0005), so 2-3 SE in every scenario.
- **Conditional prediction:** conflict given unrest improves by about 0.0005 (about 1 SE). AUC differences are within noise.

## 2026-11-07 21:05 EST: Round 9 joint prediction west_sahel done (1706562.1, rc 0, about 50 min)
`results_real/bivariate/west_sahel/summary_jointpred.json` is committed. Test set: 260,520 cell-weeks.
- **Observed:** both events in the same cell-week 0.35%. P(conflict | unrest) = 0.251, against P(conflict | no unrest) = 0.039.
- **Joint log score** (lower is better):

| model | joint log score |
|---|---|
| product of uni_static | 0.15070 |
| product of uni_st | 0.15196 |
| biv, joint draws | **0.14993** |
| biv, treated as independent | 0.15013 |

  - biv − uni_st product: −0.00202 (SE cell 0.00021, country 0.00074; about 2.7 SE).
  - biv − uni_static product: −0.00077 (country SE 0.00057).
  - biv joint − independent: −0.00020 (country SE 0.00008).
- **Conditional:** for conflict given same-week unrest, the log score is 0.10189 against 0.10306 marginal. AUC-PR is
  0.4665 against 0.4662.
- **Co-occurrence AUC-PR:** joint 0.167, independent 0.168.
- **Univariate unrest models** (new): uni_static_u AUC-PR 0.286 / log 0.0487; uni_st_u 0.284 / 0.0493.
  Bivariate unrest: 0.289 / 0.0490.
- africa_me joint prediction (1706562.2) is still running.

## 2026-11-07 23:00 EST: proposals to speed up the heavy fits (Stefano asked; these are your code, please review and push)
The biv2 fits are slow: West Africa is past 8 h, and Africa + ME is projected at 30-45 h. In order of gain per effort:

1. **Warm-start biv2 from the saved biv hyperparameters.** At present every block starts at the generic defaults
   (range 300 km, sd 0.5, rho 0.8). Suggested change in `fit_bivariate.py`:
   - add `--init-theta summary_biv.json`;
   - for the blocks whose names exist in `saved["models"]["biv"]["hyper"]`, set `theta0` via `theta_from_hyper`;
   - keep the defaults only for the new block, `st_unrest`;
   - then `m.fit(theta0=theta0)`.

   Expected gain: about 2-3× fewer L-BFGS iterations. The same applies to any refit after a code change.
2. **Looser outer tolerance.** In `LatentGaussianModel.fit`, pass `options=dict(maxiter=maxiter, ftol=1e-7, gtol=1e-3)`
   (or settings taken from env vars) and cap `maxiter` at about 60 for the big fits. With forward finite differences, the
   last iterations mostly chase gradient noise. Expected gain: about 1.5×. Please check the hyperparameters move by
   less than about 1% on a fit you already have.
3. **Pin heavy fits to 64-core EPYC nodes.** Both biv2 jobs landed on the older 24-core d12chas nodes (ACMS), which are
   noticeably slower per core. In `16/17/18_*.sh`, use `#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc`.
   Expected gain: about 1.5-2×.
4. **Coarser resolution for diagnostic runs:** 2° space-time cells or 26-week steps (about 4-8× fewer unknowns). Keep
   1° / 13 weeks for the final paper fits. In Round 4, 26 weeks gave the same answers as 13.
5. **Use `--fixed-theta` for everything after the fit** (prediction, updating, sensitivity), as 18_joint_pred already does.

**Analytic gradient (in progress on my side; not for use yet).** Branch `analytic-grad` in a separate worktree. Nothing
on main is touched.
- New `sc/selinv.py`: a Takahashi selected inverse in numba on a CHOLMOD simplicial factor.
- New `LatentGaussianModel.log_marginal_and_grad`: the exact Laplace gradient, including the implicit mode term.
  It is opt-in with `LAPLACE_GRAD=analytic`, and the default stays finite differences.
- New `tools/check_analytic_grad.py`.
- Verified against central finite differences on all block types (Fixed, IID, Matern with barrier, SpaceTimeAR1,
  CopyBlock), for Bernoulli and Poisson, with max rel.err 2e-7.
- Not yet faster: on a 26k-latent, 240k-row bivariate, one analytic gradient takes 10.1 s against 0.70 s per Laplace
  solve. 7.5 s of that is the serial Takahashi loop.
- Next: a thread-parallel selected inverse over the elimination tree. I'll report when it beats parallel finite
  differences. It needs scikit-sparse and numba, which are installed only in a separate env, `/users/scastruc/envs/conflict-dev`.

## 2026-11-07 22:45 EST: warm-start biv2 submitted (repo at 914637f)
- `19_biv2_warm.sh`: **1708086** (tasks 1-2: west_sahel, africa_me). It runs on EPYC queues, warm-started, with LAPLACE_FTOL 1e-7 /
  GTOL 1e-3 / MAXITER 60.
- **Submitted with 16 slots, not 32.** Stefano set the limit at 16 cores per batch job (2026-11-07). I used a copy of the
  script with `-pe smp 16` and `FIT_JOBS=8`, keeping OMP 2 threads per worker, so 16 threads in total. Please change
  the header in the repo to 16 / FIT_JOBS 8 so it matches.
- Deleted 1705023, the old africa_me biv2 (cold start), after 5 h 50 min. It wrote no output.
- Still running:
  - 1701825, the old west_sahel biv2, as the cross-check. Whichever finishes first is committed as `summary_biv2.json`,
    and if both finish I'll report the differences in hyperparameters and κ.
  - 1706562.2, the africa_me joint prediction.
- Analytic gradient: I'm continuing on the branch, and it won't be merged until it beats the parallel finite differences.

## 2026-11-07 23:20 EST: Round 9 joint prediction africa_me done (1706562.2, rc 0, about 2 h 40 min, finished 22:16)
`results_real/bivariate/africa_me/summary_jointpred.json` is committed. Test set: 1,426,776 cell-weeks.
- **Observed:** both events in the same cell-week 0.36%. P(conflict | unrest) = 0.135, against P(conflict | no unrest) = 0.021.
- **Joint log score** (lower is better):

| model | joint log score |
|---|---|
| product of uni_static | 0.12268 |
| product of uni_st | 0.12455 |
| biv, joint draws | **0.12168** |
| biv, treated as independent | 0.12172 |

  - biv − uni_st product: −0.00287 (SE cell 0.00011, country 0.00033; **about 8.8 SE**).
  - biv − uni_static product: −0.00101 (country SE 0.00063, about 1.6 SE).
  - biv joint − independent: −0.00004 (country SE 0.00003).
- **Conditional:** for conflict given same-week unrest, the log score is 0.05849 against 0.05950 marginal; AUC-PR 0.526.
- **Co-occurrence AUC-PR:** joint 0.416, independent 0.417.
- **Univariate unrest models:** uni_static_u AUC-PR 0.545 / log 0.0634; uni_st_u 0.532 / 0.0646.
- **Round 9 is complete:** S5 v3, joint prediction for west_sahel and for africa_me.

## 2026-11-07 23:02 EST: rescoring submitted (repo at 167e988, Monte Carlo scoring for every model)
- 18_joint_pred: **1708150** (tasks 1-2, `--fixed-theta`). It will overwrite `summary_jointpred.json` for both regions.
- S5 v4: **1708152** (40 tasks, checkpoints `s5v4_ckpt*`), with merge **1708153** held on it.
- The biv2 jobs keep running: 1708086 (warm, both regions) and 1701825 (west_sahel cold start, the cross-check).
- Thanks for the 19_biv2_warm.sh header fix (smp 16 / FIT_JOBS 8).

## 2026-11-07 23:30 EST: west_sahel biv2 (cold start, 1701825) done, and west_sahel joint prediction rescored (1708150.1)
### biv2, west_sahel: the cold start finished first
- Job 1701825: rc 0, 15:08-23:26 on d12chas216 (8 h 18 min), fit 29,652 s, no Traceback.
  `results_real/bivariate/west_sahel/summary_biv2.json` is committed. The warm run (1708086.1) is still going, and I'll
  report the differences when it finishes.
- log-ML: biv2 −76,461 against biv −76,592 (**+131**).
- Test scores: conflict AUC-PR 0.470 / log 0.1029; unrest 0.283 / 0.0496. For biv these were 0.466 / 0.1031 and 0.289 / 0.0490.

**Lag coefficients** (estimate, SE):

| term | biv | biv2 |
|---|---|---|
| conflict:lag_own | 0.459 (0.017) | 0.448 (0.017) |
| **conflict:lag_other** | **0.044 (0.032)** | **0.095 (0.032)**, about 3 SE |
| conflict:lag_own_nb | 0.061 | 0.054 |
| conflict:lag_other_nb | −0.020 | −0.005 |
| unrest:lag_own | 0.351 | 0.269 |
| unrest:lag_other | 0.066 | 0.111 (0.031) |
| unrest:lag_own_nb | 0.017 | −0.004 |
| unrest:lag_other_nb | −0.042 | −0.008 |

**biv2 hyperparameters:**
- country: conflict 0.673, unrest 0.514;
- cell: conflict 0.654, unrest 1.358;
- field_conflict: 467 km, sd 1.443;
- field_unrest: 497 km, sd 0.344;
- shared_st: 445 km, sd 0.993, rho 0.957, **loading 0.58** (biv: 0.868);
- **st_unrest: 787 km, sd 0.494, rho 0.687.**

With an unrest-specific space-time field, the shared field's loading drops, and unrest → conflict comes back to about 0.095.
This is the pattern S5 v2 predicted, where biv_st under-estimates escalation when unrest has its own episodes.

### Joint prediction, west_sahel, rescored with Monte Carlo (1708150.1, rc 0, finished 23:21)
`results_real/bivariate/west_sahel/summary_jointpred.json` is committed and overwritten.

| model | joint log score (MC) | probit version from 21:05 |
|---|---|---|
| product of uni_static | 0.14978 | 0.15070 |
| product of uni_st | 0.14991 | 0.15196 |
| biv, joint draws | 0.14993 | 0.14993 |

- **With the same Monte Carlo scoring, biv's joint-score advantage disappears in west_sahel:**
  - biv − uni_st product = +0.00003 (country SE 0.00050);
  - biv − uni_static product = +0.00015 (SE 0.00062);
  - joint − independent = −0.00020 (SE 0.00008).
- Conditional: conflict given unrest 0.10189 against marginal 0.10209.
- Co-occurrence AUC-PR: joint 0.167, independent 0.166.
- africa_me (1708150.2) is still running.

## 2026-11-07 23:54 EST: region selection submitted (repo at 6062f10)
- 20_select_region.sh: **1708445**. It runs a 2018-21 fit with 2022 validation scoring, on fixed hyperparameters, followed by joint_gain_map.
- After 1708150.2 finishes, I'll run joint_gain_map on its test parquet with the `test2324` prefix (descriptive only).
  That run was submitted at 167e988, before 6062f10 added iso/lon/lat to the prediction parquet. If its parquet lacks
  those columns, I'll say so here rather than rerun anything.
- No regional fit will be started.

## 2026-11-08 00:25 EST: region selection (1708445) ran, but the result is INVALID because of a bug in --validation; selection not committed
- Job 1708445: rc 0 at 00:11. Its files were written into results_real/bivariate/selection/ on the CRC, but I have
  **not** committed them.
- **Symptom:** on 2022, every model's calibration collapses while ranking is fine.

| model | log score | Brier | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| uni_static, conflict | 0.547 | 0.178 | 0.961 | 0.546 |
| uni_static_u, unrest | 0.546 | 0.177 | 0.959 | 0.534 |
| biv, conflict | 0.612 | 0.210 | 0.961 | 0.543 |
| biv, unrest | 0.536 | 0.172 | 0.959 | 0.535 |

  The log scores are about 10x the usual 0.06 at prevalence 2.4-2.5%, but AUC is unchanged.
  - mean_gain = −0.058 (biv − product). Every window has a negative gain; the "selected" one is lon 30-40, lat −35 to −25
    (ZAF/MOZ/SWZ), at −0.047.
- **Cause:** `fit_glmm.prepare()` builds the year dummies on the original training years:
  `yr = week_start.year.clip(upper=2022)`, `years = sorted(yr[df.train].unique())`, giving year_2019 to year_2022.
  - `fit_bivariate --validation` then sets train = before 2022, so **year_2022 is 0 on every training row and 1 only
    on the 2022 rows**.
  - Its coefficient is unidentified, so it keeps the FixedBlock vague prior (prec 1e-4, sd 100).
  - The Monte Carlo predictive draws (`mc_prob`) then add a common N(0, about 100²) shift to every 2022 eta.
  - The probabilities go to about 0 or 1 at random: calibration is destroyed and ranking is untouched.
- **Fix (suggested):**
  - In validation mode, give 2022 the 2021 year effect, mirroring the "test years take the 2022 effect" convention.
    For example, pass a `last_train_year` to `prepare` and use `clip(upper=last_train_year)`. Or, after the train
    split, drop any year dummy that is constant on the training rows.
  - The same trap applies to any covariate level absent from the training rows (biome levels are already handled).
  - A general guard: in `fit_or_condition`, warn if a fixed-effect column is all zero on the training rows.
- **On hold:** the test2324 gain map, since the window selection it was meant to accompany is void. I'll rerun 20_select_region.sh
  once you push a fix.

## 2026-11-08 00:25 EST: west_sahel biv2 warm start done (1708086.1): reproduces the cold fit about 6× faster
`results_real/bivariate_warm/west_sahel/summary_biv2.json` is committed. The canonical results_real/bivariate/west_sahel/summary_biv2.json is still the cold run.

| | cold (1701825) | warm (1708086.1) |
|---|---|---|
| fit seconds | 29,652 (d12chas, old engine settings) | **4,932** (EPYC, warm, looser tol.) |
| log-ML | −76,461.03 | −76,461.32 |

- Lag coefficients are identical to 3 decimals in both fits: conflict:lag_other 0.095 (0.032), unrest:lag_other 0.111 (0.031).
- Most hyperparameters agree within 0-3%: shared loading 0.580 against 0.590, st_unrest 787 against 791 km, rho 0.687 against 0.698.
- **Exception, the weakly identified field_unrest:** range 497 against 371 km, sd 0.344 against 0.224. The likelihood
  is flat in that direction, which is why log-ML is unchanged.
- africa_me warm (1708086.2) is still running, at 2 h 37 min.

## 2026-11-08 00:14 EST: region selection resubmitted with the year-dummy fix (f6c411d): **1708630**
- I deleted the invalid, uncommitted selection files from the first run.
- The test2324 gain map (with --lookup) follows once this job and the africa_me rescoring (1708150.2) have finished.

## 2026-11-08 00:45 EST: region selection on 2022 done (1708630, rc 0, about 15 min, no Traceback). Valid this time
- Committed `results_real/bivariate/selection/`: val2022_summary.json, val2022_by_country.csv, val2022_windows.csv and summary_val.json.
- **Dropped covariates** (log line, for both outcomes):
  `conflict: dropping covariates that are zero on all training rows: ['year_2022']`, and the same for unrest.
- **2022 scores are now properly calibrated.** Validation rows: 713,388 per outcome.

| model | conflict log / Brier / AUC-PR | unrest log / Brier / AUC-PR |
|---|---|---|
| uni_static | 0.0574 / 0.0155 / 0.546 | 0.0608 / 0.0164 / 0.535 |
| biv | 0.0569 / 0.0154 / 0.549 | 0.0606 / 0.0163 / 0.538 |

- **Joint log score on 2022:** product of uni_static 0.11823, biv 0.11752.
  - biv − product = −0.00071 (SE cell 0.00013, country 0.00042, about 1.7 SE).
  - Joint − independent = −0.00006 (country SE 0.00002).
  - mean_gain is 0.00071 and mean_gain_dependence 0.00006, so nearly all of the gain comes from better marginals, not from the dependence.
- **Selected window** (largest total gain, at least 50 cells): **lon 60-70, lat 30-40 (AFG, TKM, IRN, UZB)**.
  200 cells, gain_sum 180.3, gain_mean 0.0173 (SE country 0.0117), dep_mean −0.00002.
- **Top 5 windows on 2022:**

| lon | lat | cells | gain_sum | gain_mean | SE country | dep_mean | both | countries |
|---|---|---|---|---|---|---|---|---|
| 60-70 | 30-40 | 200 | 180.3 | 0.0173 | 0.0117 | −0.00002 | 0.0013 | AFG, TKM, IRN, UZB |
| 55-65 | 30-40 | 400 | 173.0 | 0.0083 | 0.0090 | −0.00001 | 0.0010 | IRN, TKM, AFG, UZB |
| 0-10 | 30-40 | 278 | 167.4 | 0.0116 | 0.0017 | 0.00002 | 0.0001 | DZA, TUN, ITA, ESP, LBY |
| 60-70 | 25-35 | 194 | 143.8 | 0.0143 | 0.0101 | 0.00004 | 0.0040 | AFG, IRN, PAK |
| 55-65 | 25-35 | 375 | 125.1 | 0.0064 | 0.0083 | 0.00002 | 0.0021 | IRN, AFG, PAK, OMN, ARE |

- **Top countries, 2022:**

| country | cells | gain_sum | gain_mean |
|---|---|---|---|
| AFG | 104 | 173.1 | 0.0320 |
| DZA | 854 | 171.5 | 0.0039 |
| IRQ | 170 | 58.7 | 0.0066 |
| NGA | 300 | 43.5 | 0.0028 |
| GIN | 78 | 39.5 | 0.0097 |
| SSD | 202 | 38.7 | 0.0037 |

- **Note:** in every top window, dep_mean is about 0. The gains there come from biv's better marginal fit, through the
  shared field and country effects, not from modelling the co-occurrence. Most top windows also have very few joint
  events (both ≤ 0.4%).
- The test2324 gain map follows once 1708150.2 finishes. It is still running, about 1 h 50 min in.

## 2026-11-08 00:52 EST: S5 v4 done (1708152, 40 tasks, no Tracebacks; merge 1708153 rc 0). All prediction scores now use Monte Carlo
Committed `results/s5_raw.csv`, `s5_summary.csv` (estimates unchanged) and `s5_prediction.csv`:
```
         scenario   model  n  joint_log_score  joint_log_score_indep  cond_log_score  cond_auc_pr  both_auc_pr  both_auc_pr_indep  joint_log_score_minus_uni  joint_log_score_minus_uni_se  cond_log_score_minus_uni  cond_log_score_m
       escalation  biv_st 20           0.2890                 0.2892          0.1394       0.1038       0.0275             0.0273                     0.0001                        0.0005                    0.0001                  
       escalation biv_st2 20           0.2889                 0.2891          0.1392       0.1052       0.0275             0.0277                    -0.0001                        0.0005                   -0.0001                  
       escalation  uni_st 20           0.2889                    NaN          0.1393       0.1049       0.0324                NaN                     0.0000                        0.0000                    0.0000                  
   escalation_uST  biv_st 20           0.3063                 0.3064          0.1399       0.1044       0.0321             0.0329                    -0.0001                        0.0005                    0.0001                  
   escalation_uST biv_st2 20           0.3063                 0.3065          0.1397       0.1062       0.0325             0.0333                    -0.0001                        0.0005                   -0.0001                  
   escalation_uST  uni_st 20           0.3064                    NaN          0.1398       0.1073       0.0359                NaN                     0.0000                        0.0000                    0.0000                  
    no_escalation  biv_st 20           0.2832                 0.2834          0.1335       0.0884       0.0242             0.0240                    -0.0001                        0.0005                   -0.0001                  
    no_escalation biv_st2 20           0.2831                 0.2833          0.1334       0.0897       0.0249             0.0243                    -0.0002                        0.0005                   -0.0002                  
    no_escalation  uni_st 20           0.2833                    NaN          0.1336       0.0900       0.0273                NaN                     0.0000                        0.0000                    0.0000                  
no_escalation_uST  biv_st 20           0.3000                 0.3002          0.1336       0.0882       0.0273             0.0269                    -0.0003                        0.0005                   -0.0001                  
no_escalation_uST biv_st2 20           0.3002                 0.3004          0.1336       0.0892       0.0260             0.0266                    -0.0002                        0.0005                   -0.0001                  
no_escalation_uST  uni_st 20           0.3003                    NaN          0.1337       0.0897       0.0294                NaN                     0.0000                        0.0000                    0.0000                  
rc=0
```
- With identical Monte Carlo scoring, **biv_st and biv_st2 no longer beat uni_st on the joint or conditional log score**
  in any scenario. The differences are −0.0003 to +0.0001 (SE 0.0005); in S5 v3 they were about −0.001.
- Joint against independent draws is about −0.0002 in every case. The S5 v3 joint gain was the same scoring artefact as
  in the application.

## 2026-11-08 02:00 EST: africa_me joint prediction rescored with Monte Carlo (1708150.2, rc 0, 01:50) and 2023-24 gain map (1709550, rc 0)
### Rescored africa_me joint prediction
`results_real/bivariate/africa_me/summary_jointpred.json` is committed and overwritten. Test set: 1,426,776 cell-weeks.

| model | joint log score (MC) | probit version from 23:20 |
|---|---|---|
| product of uni_static | 0.12170 | 0.12268 |
| product of uni_st | 0.12257 | 0.12455 |
| biv | 0.12168 | 0.12168 |

- **biv − uni_static product = −0.00002 (country SE 0.00056): no gain.** The earlier −0.00101 came from the probit
  approximation.
- biv − uni_st product = −0.00089 (country SE 0.00032, about 2.7 SE). The earlier figure was −0.00287.
- Joint − independent = −0.00004 (country SE 0.00003).
- Conditional: conflict given unrest 0.05849 against marginal 0.05853. Co-occurrence AUC-PR: 0.416 joint, 0.417 independent.

### Gain map on 2023-24, descriptive (`results_real/bivariate/selection/test2324_*`)
- `location lookup: 0 rows without a match`.
- Overall: mean_gain = 0.00002 (biv over the uni_static product), mean_gain_dependence = 0.00004.
- **The top 5 windows selected on 2022, evaluated on 2023-24:**

| lon | lat | countries | cells | gain_mean | SE country | dep_mean | both |
|---|---|---|---|---|---|---|---|
| 60-70 | 30-40 | AFG, TKM, IRN, UZB | 200 | **0.0148** | 0.0110 | 0.00009 | 0.0008 |
| 55-65 | 30-40 | IRN, TKM, AFG, UZB | 400 | 0.0073 | 0.0082 | 0.00004 | 0.0005 |
| 0-10 | 30-40 | DZA, TUN, ITA, ESP, LBY | 278 | 0.0103 | 0.0018 | −0.00002 | 0.0001 |
| 60-70 | 25-35 | AFG, IRN, PAK | 194 | 0.0155 | 0.0080 | 0.00032 | 0.0054 |
| 55-65 | 25-35 | IRN, AFG, PAK, OMN, ARE | 375 | 0.0077 | 0.0073 | 0.00016 | 0.0028 |

  The selected window keeps its 2022 sign and size in 2023-24: 0.0173 on 2022, 0.0148 on 2023-24. But the uncertainty
  is wide in four of the five, at about 1-2 SE (country-clustered). Only the Maghreb window is precise. dep_mean is
  about 0 in every window, so the gain is in the marginals, not the dependence.
- **Top countries in 2023-24, by gain_sum:**

| country | cells | gain_sum | gain_mean | dep_mean | both |
|---|---|---|---|---|---|
| AFG | 104 | 310.4 | 0.0287 | | |
| DZA | 854 | 307.6 | 0.0035 | | |
| NGA | 300 | 178.2 | 0.0057 | 0.0003 | 0.017 |
| COD | 759 | 120.9 | 0.0015 | | |
| TUR | 324 | 89.2 | 0.0026 | | |
| KEN | 185 | 86.6 | 0.0045 | | |
| BFA | 86 | 80.6 | 0.0090 | 0.0007 | 0.011 |
| LBY | 592 | 55.5 | 0.0009 | | |

  AFG and DZA also topped 2022.

## 2026-11-08 03:35 EST: africa_me biv2, warm start, done (1708086.2, rc 0, 22:48-03:30, **4 h 42 min**; fit 16,742 s; no Traceback)
- This was the only africa_me biv2 run (the cold one, 1705023, was deleted). Committed as
  `results_real/bivariate/africa_me/summary_biv2.json`, with a copy in `results_real/bivariate_warm/africa_me/`.
- Setup: 26-week space-time steps, as in the job script. biv had v on 14 steps.
- log-ML: biv2 −353,977 against biv −354,669 (**+692**).
- Test scores: conflict AUC-PR 0.525 / log 0.0597; unrest 0.534 / 0.0646.

**Lag coefficients** (estimate, SE):

| term | biv | biv2 |
|---|---|---|
| **conflict:lag_other** | **0.057 (0.015)** | **0.105 (0.015)**, about 7 SE |
| unrest:lag_other | 0.025 (0.014) | 0.074 (0.014) |
| conflict:lag_own | 0.626 | 0.608 |
| conflict:lag_own_nb | 0.083 | 0.073 |
| conflict:lag_other_nb | −0.035 | −0.015 |
| unrest:lag_own | 0.585 | 0.542 |
| unrest:lag_own_nb | 0.071 | 0.057 |
| unrest:lag_other_nb | −0.042 | −0.008 |

**biv2 hyperparameters:**
- country sd: conflict 0.519, unrest 0.925;
- cell sd: conflict 0.987, unrest 1.318;
- field_conflict: 598 km, sd 1.214;
- field_unrest: 417 km, sd 0.476;
- shared_st: 472 km, sd 1.260, rho 0.948, **loading 0.44** (biv: 0.668);
- **st_unrest: 697 km, sd 0.429, rho 0.696.**

Same pattern as west_sahel. With an unrest-specific space-time field, the shared loading falls and unrest → conflict is
about 0.10 in both regions (west_sahel 0.095 (0.032), africa_me 0.105 (0.015)).

All conflict jobs are now finished. Only my analytic-gradient comparison (1708140) is still running.

## 2026-11-08 13:00 EST: new CRC session; 1708140 status; dependence diagnostic on 2023-24 (1711662, rc 0)
### 1708140 (analytic vs finite-difference fit comparison, branch `analytic-grad`, not merged): **still running**
- Script `~/conflict_check/fitcmp.sh`; log `~/conflict_check/fitcmp.log`. Test: `tools/compare_fit.py`, 80×60 grid, T=14,
  4 reps, bivariate. Analytic first (OMP 16), then finite differences (FIT_JOBS 8 × OMP 2).
- **Analytic half done:** wall 1,358 s, 48 L-BFGS iterations / 56 evaluations, log-ML −180,888.63, converged.
  The hyperparameters look degenerate: c1.sd 0.012, c2.sd 0.022, shared range 27 km, rho 0.995,
  loading 0.9996. Check whether the FD fit lands at the same point before reading anything into the speed.
- **FD half:** started about 23:26 and has written nothing in roughly 13 h. qstat: cpu 5.5 h over 13.3 h wall on 16 slots,
  so it is mostly idle (perhaps stuck in the worker pool). h_rt is 24 h, so it is killed at about 23:03 tonight if it
  doesn't finish. **No speed comparison exists yet.** I'll report when it ends.

### Dependence diagnostic, 2023-24 test parquets (no refit)
`realdata/dependence_check.py`, job `crc/jobs/21_dependence.sh` (1711662, rc 0); output
`results_real/bivariate/selection/dependence_check.json`. The inputs are the MC-rescored `test_pred_jointpred.parquet` files (west_sahel
= West Africa and Sahel, africa_me = AME). Model: logit y_conflict ~ logit(p_conflict_model) + x, where x is either y_unrest
of the same cell-week or the cell's y_unrest the week before (first test week dropped). SEs are country-clustered;
Yemen and Sudan are clustered by cell.

**(a) Co-occurrence rate, mean(y_c·y_u), against the forecasts**

| subset | observed | biv P(both) | biv pc·pu | uni_static pc·pu | uni_st pc·pu |
|---|---|---|---|---|---|
| AME | 0.00363 | 0.00333 | 0.00302 | 0.00335 | 0.00326 |
| West Africa | 0.00352 | 0.00273 | 0.00227 | 0.00346 | 0.00279 |
| AFG window | 0.00082 | 0.00053 | 0.00043 | 0.00086 | 0.00096 |
| Maghreb window | 0.00014 | 0.00060 | 0.00041 | 0.00055 | 0.00041 |
| Yemen | 0.0597 | 0.0307 | 0.0285 | 0.0399 | 0.0305 |
| Sudan | 0.00159 | 0.00275 | 0.00250 | 0.00291 | 0.00267 |

**(b, c) Odds ratio for unrest, conditional on the model's conflict forecast** (95% CI; n_both = joint events)

| subset | model | same-week y_unrest | last-week y_unrest |
|---|---|---|---|
| AME (93 countries, n_both 5,173) | biv | **1.32** [1.15, 1.52] | 1.07 [0.93, 1.24] |
| | uni_static | 1.19 [1.01, 1.41] | 0.92 [0.79, 1.09] |
| | uni_st | 1.14 [0.98, 1.33] | 0.90 [0.77, 1.06] |
| West Africa (22 countries, n_both 918) | biv | **1.69** [1.41, 2.04] | **1.33** [1.18, 1.49] |
| | uni_static | 1.44 [1.16, 1.78] | 1.07 [0.95, 1.19] |
| | uni_st | 1.46 [1.23, 1.73] | 1.09 [0.96, 1.25] |
| AFG window (4 countries, n_both 17) | biv | 4.0 [2.8, 5.8] | 3.4 [1.6, 7.0] |
| | uni_static | 4.4 [2.6, 7.2] | 3.3 [1.3, 8.5] |
| | uni_st | 4.3 [3.0, 6.0] | 3.4 [1.7, 7.0] |
| Maghreb window (5 countries, n_both 4) | biv | 0.53 [0.12, 2.4] | 0.22 [0.06, 0.89] |
| | uni_static | 0.97 [0.38, 2.5] | 0.38 [0.17, 0.84] |
| | uni_st | 0.40 [0.06, 2.7] | 0.16 [0.03, 0.89] |
| Yemen (155 cells, n_both 963) | biv | **1.54** [1.20, 1.99] | 1.25 [0.98, 1.58] |
| | uni_static | 1.17 [0.92, 1.49] | 0.90 [0.72, 1.12] |
| | uni_st | 1.13 [0.88, 1.46] | 0.88 [0.70, 1.12] |
| Sudan (628 cells, n_both 104) | biv | 1.87 [1.23, 2.83] | 1.31 [0.96, 1.80] |
| | uni_static | 2.01 [1.31, 3.06] | 1.33 [0.96, 1.83] |
| | uni_st | **2.21** [1.52, 3.21] | **1.54** [1.17, 2.04] |

What the numbers show:
- **Same-week dependence remains after each model's conflict forecast:** OR about 1.15-1.3 in AME and 1.4-1.7 in West
  Africa. Observed co-occurrence is above biv's P(both) (AME +9%, West Africa +29%, Yemen about 2×). The univariate
  product is no worse overall: uni_static pc·pu is close to the observed rate in both regions. biv's marginals
  under-predict joint events.
- **Last week's unrest adds little to the univariate forecasts** (OR 0.9-1.1, not significant in AME or West Africa). Sudan
  is the exception, with uni_st 1.54. So the lag_other signal is mostly absorbed already.
- **Unexpected:** the residual ORs are *larger* for biv than for the univariate models (West Africa last-week 1.33 against
  1.07; Yemen same-week 1.54 against 1.17). biv's conflict marginal seems less sharp where unrest is high, even though it has
  lag_other. Worth checking how biv's test-time conflict forecast uses the observed lagged unrest.
- **The windows are not informative:** AFG has 17 joint events and only 4 country clusters, and Maghreb has 4 joint events. The
  large AFG odds ratios (about 4) are consistent across models, but their SEs rest on 4 clusters and should not be relied on.

## 2026-11-08 13:20 EST: submitted 22_zerofrac and 23_holdout (repo at e8a5f6b)
- `22_zerofrac.sh`: **1711777** (tasks 1-4: west_sahel biv r=0.10, west_sahel biv r=0.25, africa_me biv r=0.10,
  africa_me biv2 r=0.10, cold). 16 slots, EPYC queues.
- `23_holdout.sh`: **1711778** (tasks 1-5 west_sahel folds 0-4, tasks 6-10 africa_me folds 0-4). 16 slots, EPYC queues.
- **1708140 (FD half stuck): not yet killed.** Claude Code's automatic safety check blocked my `qdel 1708140`, so it needs the owner to run
  `qdel 1708140`, or to allow it. Otherwise it ends at h_rt (about 23:03 tonight). The analytic result above stands. The FD half has
  written no output since 23:26 yesterday.

## 2026-11-08 14:00 EST: submitted 24_biv3 (repo at 62e6542)
- `24_biv3.sh`: **1711845** (tasks 1-2: west_sahel, africa_me). 16 slots, EPYC queues.
- 1708140: (correction) the "owner will qdel" note came from the coordinating session, not the owner. Killed by me at the owner's request at 15:20 EST, see below.

## 2026-11-08 14:45 EST: country hold-out done (1711778, tasks 1-10, no Traceback); folds merged
Merged with `holdout_check.py --merge`, giving `results_real/bivariate/holdout/{west_sahel,africa_me}.json`. Conflict is treated as unobserved in
held-out countries, and each model is re-conditioned at its saved hyperparameters. The tables give mean log-score differences
(negative = better than the baseline), with country-clustered SEs in parentheses.

**West Africa and Sahel** (past: 629,778 rows / 18,641 events; test 2023-24: 253,864 / 10,994)

| | past vs uni_static | past vs uni_st | test vs uni_static | test vs uni_st |
|---|---|---|---|---|
| p_biv | −0.00268 (0.00181) | **−0.00379 (0.00162)** | +0.00077 (0.00237) | −0.00390 (0.00230) |
| pcg_biv | −0.00271 (0.00182) | **−0.00382 (0.00163)** | +0.00053 (0.00237) | −0.00414 (0.00228) |
| p_biv2 | −0.00005 (0.00177) | −0.00116 (0.00168) | +0.00156 (0.00281) | **−0.00311 (0.00155)** |
| pcg_biv2 | −0.00007 (0.00176) | −0.00118 (0.00167) | +0.00141 (0.00282) | **−0.00327 (0.00155)** |

**Africa-Middle East** (past: 3,198,054 / 77,483; test: 1,293,032 / 34,406)

| | past vs uni_static | past vs uni_st | test vs uni_static | test vs uni_st |
|---|---|---|---|---|
| p_biv | −0.00229 (0.00122) | −0.00166 (0.00115) | −0.00182 (0.00145) | −0.00097 (0.00153) |
| pcg_biv | −0.00231 (0.00122) | −0.00168 (0.00115) | −0.00192 (0.00144) | −0.00107 (0.00155) |
| p_biv2 | **−0.00219 (0.00110)** | −0.00156 (0.00100) | −0.00151 (0.00109) | −0.00065 (0.00107) |
| pcg_biv2 | **−0.00220 (0.00110)** | −0.00157 (0.00100) | −0.00163 (0.00111) | −0.00077 (0.00110) |

What the numbers show:
- **In held-out countries the joint models do better than the univariate ones in AME, in both periods:** about −0.002 against uni_static, at
  roughly 1.3-2 SE. The sign is consistent, but no single comparison passes 2 SE except the AME past comparison of biv2 against uni_static.
- **In West Africa the result depends on the period.** biv beats uni_st in the past (2.3 SE) and in the test period (1.7 SE). Against uni_static
  there is a gain in the past (1.5 SE) and none in the test period (+0.0008 ± 0.0024).
- **Conditioning on same-week unrest (pcg) adds very little over the marginal p:** at most 0.0003.
- West Africa test: uni_static beats uni_st by 0.0047 (1.7 SE). The uni_st comparisons there are against a weak baseline.

## 2026-11-08 15:30 EST: zero-frac speed-up, West Africa biv done (1711777.1-2, no Traceback); AME tasks 3-4 and biv3 (1711845) running
Summaries: `results_real/bivariate_zf/west_sahel_biv_r{0.10,0.25}/summary_biv.json`.

| | full biv | r = 0.10 | r = 0.25 |
|---|---|---|---|
| wall (s) | 16,253 | **2,691** (hyper 2,538) | 2,915 (hyper 2,759) |
| speed-up | | **6.0×** | 5.6× |
| subsample rows (of 1,292,580) | | 153,740 | 343,407 |
| L-BFGS iterations / evaluations | | 47 / 754 | 51 / 767 |
| log-ML (full data, at these θ) | −76,591.8 | −76,600.4 | −76,592.4 |
| **κ = conflict:lag_other** | 0.044 (0.032) | 0.055 (0.032) | 0.047 (0.032) |
| unrest:lag_other | 0.066 (0.031) | 0.080 (0.031) | 0.069 (0.031) |
| test conflict log / AUC-PR | 0.10209 / 0.4643* | 0.10196 / 0.4653 | 0.10203 / 0.4646 |
| test unrest log / AUC-PR | 0.04805 / 0.2864* | 0.04786 / 0.2885 | 0.04803 / 0.2864 |

\* Full-biv test scores are from the MC rescoring in summary_jointpred.json (1708150.1), the same scoring as the zf runs.
The older summary_biv.json values (0.10306 / 0.04899) used the earlier approximation and are not comparable.

Hyperparameters, in order full / r=0.10 / r=0.25:
- country sd: conflict 0.691 / 0.724 / 0.709; unrest 0.528 / 0.556 / 0.550.
- cell sd: conflict 0.661 / 0.675 / 0.660; unrest 1.335 / 1.353 / 1.341.
- field_conflict: range 453 / 445 / 453 km, sd 1.387 / 1.355 / 1.382.
- field_unrest: range 369 / 317 / 322 km, sd 0.213 / 0.176 / 0.179.
- shared_st: range 499 / 603 / 531 km, sd 1.010 / 1.045 / 1.004, rho 0.952 / 0.961 / 0.954, loading 0.868 / 0.765 / 0.875.

What the numbers show:
- **r = 0.25 essentially reproduces the full fit:** log-ML within 0.6, every lag coefficient within 0.1 SE, and the same test scores.
- **r = 0.10 is 6× faster and gives the same test scores,** but its hyperparameters drift slightly: log-ML −8.6, the shared loading
  0.77 against 0.87, and κ 0.055 against 0.044, about 0.3 SE. Both runs are fine for prediction. For reported hyperparameters, r = 0.25 is safer.
- Most of the wall time is now the hyperparameter search (95%). The latent pass on the full data takes about 150 s.

## 2026-11-08 15:20 EST: 1708140 killed (qdel, owner's request)
- The analytic half finished (1,358 s, log-ML −180,888.63; hyperparameters look degenerate, see 13:00 entry). The FD half was
  stuck: no output from 23:26 to 15:20, low CPU. **No analytic-vs-FD comparison was obtained.** Branch `analytic-grad` is untouched
  and not merged. To redo it, run the FD half on its own with a timeout, and check first that it isn't hanging in the worker pool.

## 2026-11-08 16:15 EST: submitted 25_holdout_refit (repo at 1e41294)
- `25_holdout_refit.sh`: **1712445** (20 tasks; west_sahel and africa_me × conflict and unrest targets × folds 0-4; per-fold refits with
  --zero-frac 0.25). 16 slots, EPYC queues. `26_s6.sh` is on hold, waiting for the go-ahead.
- 1708140 was killed at 15:20 (see above).

## 2026-11-08 16:30 EST: submitted 26_s6 (repo at c174e48); full-data hold-out check queued
- `26_s6.sh`: **1712461** (40 shards, 4 slots each). Submitted with `-q long@@crc_d32cepyc,long@@acms_d32cepyc`, overriding the
  script's plain `long`. Merge and commit of s6_summary.csv and s6_raw.csv follow when all shards are done.
- New `crc/jobs/27_holdout_full_check.sh`: africa_me, target conflict, fold 0, with `--zero-frac 1.0` (all rows, offset 0) and otherwise the
  same arguments as 1712445.11. Output in `$WORK/holdout_refit_full/africa_me_conflict/fold0`. It will be submitted when 1712445.11 has
  finished, and I'll add its job ID here.
