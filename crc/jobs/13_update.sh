#!/bin/bash
#$ -N cs_update
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Quarterly updating of the latent models (realdata/fit_dynamic.py --mode update):
# hyperparameters fitted on 2018-2022, latent effects recomputed each quarter of 2023-2024
# from all rows before the quarter. Task 1 = conflict, 2 = unrest.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
S=$WORK/spatial/$OUT
python realdata/fit_dynamic.py "$CLEAN" --outcome "$OUT" --region africa_me --land "$COUNTRIES" \
    --mode update --offset "$S/offset.npy" --step 13 --st-res 1.0 --nthread 4 --out "$WORK/dynamic/$OUT"
