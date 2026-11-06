#!/bin/bash
#$ -N cs_spatial
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Hybrid model (cross-fitted XGBoost offset + country effects + barrier Matern field),
# Africa + Middle East; the offset is trained on the Africa + Middle East rows only.
# Task 1 = conflict, 2 = unrest.
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
python realdata/fit_acled.py "$CLEAN" --outcome "$OUT" --region africa_me \
    --land "$COUNTRIES" --nthread 4 --out "$WORK/spatial/$OUT"
