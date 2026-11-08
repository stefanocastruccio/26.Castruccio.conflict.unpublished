#!/bin/bash
#$ -N cs_s6sf
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-40
# S6 add-on: the joint model with a shared STATIC field (biv_sf = biv_st + shared Matern field
# with a loading), i.e. the generator's structure. Same replicates, blocks and targets as 26_s6.
# Separate checkpoint s6sf_ckpt.jsonl. About 2 min per model fit, 3 fits per task.
# Afterwards merge once without SHARD (needs results/s6_raw.csv from 26_s6):
#   cd sims && N_JOBS=1 S6_MODELS=biv_sf S6_TAG=_sf S6_CKPT=s6sf_ckpt.jsonl python s6_holdout.py
# -> results/s6_sf_raw.csv, results/s6_sf_summary.csv
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export FIT_JOBS=4 OMP_NUM_THREADS=1 S6_MODELS=biv_sf S6_TAG=_sf S6_CKPT=s6sf_ckpt.jsonl
SHARD="$((SGE_TASK_ID - 1))/40" N_JOBS=1 python s6_holdout.py
