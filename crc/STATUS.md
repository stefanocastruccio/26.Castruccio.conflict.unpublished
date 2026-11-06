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
