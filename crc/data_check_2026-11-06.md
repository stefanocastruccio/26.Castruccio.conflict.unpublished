# CRC data check (step 1), 2026-11-06

Written by the Claude session on the Notre Dame CRC front-end (crcfe01). That
session can't send messages to other sessions, so it relays replies through
this repo, in files under `crc/`. Reply by committing a file here or editing
this one. Only aggregate counts are below; there are no raw ACLED rows.

## a. Data location
- Path: `/groups/scastruc/lmenical/drew/data/partitions/`
- 50 files, `data_final_world_01.csv` … `_50.csv`, 16 GB in total (file 25 is 334 MB). CSV only, no Parquet.
- There is no pipeline `results/` folder under `/groups/scastruc` (no `plot_data_*`, `shap_XGB_*` or `results/boot`). The other folders there, `lmenical/SAI_Fellowship` (258 GB) and `/groups/scastruc/data` (owner hbi), belong to other projects.
- Header (34 columns): week_start, grid_id, Protests, Battles, Strategic_developments, ExplosionsRemote_violence, Violence_against_civilians, Riots, population, gdp, phase_1, phase_2, phase_3, phase_4, phase_5, river_length, lake_area, drought, malaria, electricity_access, tmin, tmax, hi, wbgt, ppt, pm25, lon, lat, EPI_index, BIOME_NAME, GHG_pc_t, MF_pc_t, ECO_pc_gha, v2x_libdem

## b. `data_final_world_25.csv` (pandas, run as a 4-core batch job)
- 1,357,440 rows; 94,730 unique grid_id; 7 unique week_start (2021-05-11 to 2021-06-22). **The partitions split the data by time, not by space.**
- `groupby(['grid_id','week_start']).size().value_counts()`, given as rows per pair: number of pairs:
  1: 613,459 · 16: 17,542 · 8: 11,389 · 2: 11,284 · 3: 2,541 · 32: 2,219 · 48: 1,106 · 4: 777 · 81: 749 · 54: 644 · 24: 280 · 64: 203 · 27: 182 · 108: 154 · 162: 119 · 5: 98 · 243: 98 · 80: 56 · 36: 35 · 324: 28 · 6, 40: 21 each · 96, 144, 128: 14 each · 7, 648, 12, 160, 112, 216, 256, 512, 18: 7 each
- About 9% of the about 663k (cell, week) pairs are duplicated, and they hold more than half of all rows. Multiplicities such as 2^k, 3^k and 2·3^k suggest a **many-to-many merge** (several polygons per cell, e.g. for biome, EPI or country attributes) rather than simple border duplication. Deduplicate before modelling.
- **There is no country code column** (no ISO_A3, ISO*, COUNTRY*, ADM* or GID*).

## c. Environment
- Conda env `/users/scastruc/envs/geo`: Python 3.12.16, pandas 3.1.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, torch 2.15.
- **Not installed:** xgboost, pyarrow, pymetis (and gpboost). All can be installed with pip; nothing was installed.
- Scheduler: UGE/SGE `qsub`. Queues: `hpc`, `long`, `largemem`, `gpu`.
  - Working options: `-q long -pe smp 4 -l h_rt=...`. CRC rule: no job over 4 cores, and no compute on the 1-core front-end.
  - Each user can run at most 100 jobs at once (`maxujobs`).
  - qsub needs a Kerberos ticket (`KRB5CCNAME=FILE:/tmp/krb5cc_229669_persist`).
