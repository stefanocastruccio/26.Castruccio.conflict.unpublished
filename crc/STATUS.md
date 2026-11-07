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
