#!/bin/bash
#$ -N cs_spatial
#$ -q largemem
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -t 1-2
# Hybrid model (cross-fitted XGBoost offset + country effects + barrier Matern field),
# Africa + Middle East. Task 1 = conflict, 2 = unrest.
source "$(dirname "$0")/env.sh"
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
python realdata/fit_acled.py "$WORK/clean.parquet" --outcome "$OUT" --region africa_me \
    --nthread 4 --out "$WORK/spatial/$OUT"
