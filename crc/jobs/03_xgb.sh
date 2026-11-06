#!/bin/bash
#$ -N cs_xgb
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-8
# Task array (crc/TASKS.md, Round 3):
#   1-2 original  (orig.parquet, published procedure; also scored on one row per cell-week
#                  and on the cell-weeks of the repaired test set)
#   3-4 dedup     ($CLEAN, published procedure unchanged)
#   5-6 corrected ($CLEAN, no class weight, early stopping on 2022, lagged features)
#   7-8 corrected + target encoding (corrected_te)
# odd = Conflict, even = Unrest
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
MODES=(original original dedup dedup corrected corrected corrected corrected)
OUTS=(Conflict Unrest Conflict Unrest Conflict Unrest Conflict Unrest)
i=$((SGE_TASK_ID - 1)); MODE=${MODES[$i]}; OUT=${OUTS[$i]}
EXTRA=()
[ "$SGE_TASK_ID" -ge 7 ] && EXTRA=(--target-encode)
TABLE=$CLEAN
if [ "$MODE" = original ]; then
    TABLE=$WORK/orig.parquet; EXTRA=(--restrict-to "$CLEAN")
fi
python realdata/reproduce_xgb.py "$TABLE" --mode "$MODE" --outcome "$OUT" \
    --device cuda --nthread 4 --out "$WORK/xgb" "${EXTRA[@]}"
