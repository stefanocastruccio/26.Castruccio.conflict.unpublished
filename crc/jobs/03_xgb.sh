#!/bin/bash
#$ -N cs_xgb
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-6
# Task array: 1-2 original (Conflict, Unrest), 3-4 corrected, 5-6 corrected + target encoding
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
MODES=(original original corrected corrected corrected corrected); OUTS=(Conflict Unrest Conflict Unrest Conflict Unrest)
TE=("" "" "" "" --target-encode --target-encode)
i=$((SGE_TASK_ID - 1)); MODE=${MODES[$i]}; OUT=${OUTS[$i]}
TABLE=$WORK/clean.parquet; [ "$MODE" = original ] && TABLE=$WORK/orig.parquet
python realdata/reproduce_xgb.py "$TABLE" --mode "$MODE" --outcome "$OUT" \
    --device cuda --nthread 4 --out "$WORK/xgb" ${TE[$i]}
