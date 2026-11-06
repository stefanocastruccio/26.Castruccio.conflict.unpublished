#!/bin/bash
#$ -N cs_dynamic
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-8
# Noise or moving structure? (realdata/fit_dynamic.py). Needs the Round 3 spatial outputs
# ($WORK/spatial/<outcome>/offset.npy and field.csv) from 04_spatial.sh.
#   tasks 1-2  oracle     latent models fitted to the 2023-2024 test period (upper bound)
#   tasks 3-4  rolling    everything retrained on 2018-2023, scored on 2024
#   tasks 5-6  spacetime  offset + u + iid cells + AR(1) x Matern v (13-week steps, 1 degree)
#   tasks 7-8  spacetime  same with 26-week steps
# odd task = conflict, even task = unrest
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$(( (SGE_TASK_ID - 1) % 2 ))]}
MODES=(oracle oracle rolling rolling spacetime spacetime spacetime spacetime)
STEPS=(13 13 13 13 13 13 26 26)
MODE=${MODES[$((SGE_TASK_ID - 1))]}; STEP=${STEPS[$((SGE_TASK_ID - 1))]}
S=$WORK/spatial/$OUT
python realdata/fit_dynamic.py "$CLEAN" --outcome "$OUT" --region africa_me --land "$COUNTRIES" \
    --mode "$MODE" --offset "$S/offset.npy" --static-field "$S/field.csv" \
    --step "$STEP" --st-res 1.0 --nthread 4 --out "$WORK/dynamic/$OUT"
