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
