#!/bin/bash
#$ -N cs_global
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -l h_rt=168:00:00
#$ -t 1-2
# Hybrid model on the whole world (no --region).
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
python realdata/fit_acled.py "$WORK/clean.parquet" --outcome "$OUT" \
    --land "$COUNTRIES" --nthread 4 --out "$WORK/spatial_global/$OUT"
