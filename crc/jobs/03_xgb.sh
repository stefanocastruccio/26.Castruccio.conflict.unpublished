#!/bin/bash
#$ -N cs_xgb
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -t 1-4
# Task array: 1 original/Conflict, 2 original/Unrest, 3 corrected/Conflict, 4 corrected/Unrest
source "$(dirname "$0")/env.sh"
cd "$REPO"
MODES=(original original corrected corrected); OUTS=(Conflict Unrest Conflict Unrest)
i=$((SGE_TASK_ID - 1)); MODE=${MODES[$i]}; OUT=${OUTS[$i]}
TABLE=$WORK/clean.parquet; [ "$MODE" = original ] && TABLE=$WORK/orig.parquet
python realdata/reproduce_xgb.py "$TABLE" --mode "$MODE" --outcome "$OUT" \
    --device cuda --nthread 4 --out "$WORK/xgb"
