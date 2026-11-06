#!/bin/bash
#$ -N cs_compare
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -l h_rt=04:00:00
#$ -t 1-2
# Like-for-like comparison on the same Africa+ME test rows (needs Task 3 and the rerun of Task 4).
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
OUTC=${OUT^}   # Conflict / Unrest
mkdir -p "$WORK/compare"
python realdata/compare.py --xgb "$WORK/xgb/pred_corrected_${OUTC}.parquet" \
    --hybrid "$WORK/spatial/$OUT/test_pred.parquet" --out "$WORK/compare/compare_$OUT.json"
