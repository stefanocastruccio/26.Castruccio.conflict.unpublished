#!/bin/bash
#$ -N cs_xgb
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-4
# Task array: 1 original/Conflict, 2 original/Unrest, 3 corrected/Conflict, 4 corrected/Unrest
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
MODES=(original original corrected corrected); OUTS=(Conflict Unrest Conflict Unrest)
i=$((SGE_TASK_ID - 1)); MODE=${MODES[$i]}; OUT=${OUTS[$i]}
TABLE=$WORK/clean.parquet; [ "$MODE" = original ] && TABLE=$WORK/orig.parquet
python realdata/reproduce_xgb.py "$TABLE" --mode "$MODE" --outcome "$OUT" \
    --device cuda --nthread 4 --out "$WORK/xgb"
