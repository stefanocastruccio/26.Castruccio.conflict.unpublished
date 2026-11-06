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
S=$WORK/spatial/$OUT; XG=$WORK/xgb
python realdata/compare.py \
    --pred xgb=$XG/pred_corrected_${OUTC}.parquet:logit_cal \
    --pred xgb_te=$XG/pred_corrected_te_${OUTC}.parquet:logit_cal \
    --pred xgb_crossfit=$S/test_pred.parquet:p_xgb_crossfit \
    --pred hybrid_iid=$S/test_pred.parquet:p_iid \
    --pred hybrid_field=$S/test_pred.parquet:p_field \
    --lag $S/test_pred.parquet --ref xgb --out "$WORK/compare/compare_$OUT.json"
