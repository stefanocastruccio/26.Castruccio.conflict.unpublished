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
