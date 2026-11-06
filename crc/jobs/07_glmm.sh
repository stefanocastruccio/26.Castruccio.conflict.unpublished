#!/bin/bash
#$ -N cs_glmm
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -l h_rt=168:00:00
#$ -t 1-2
# Parametric models (Africa + Middle East): lags, covariates, national indicators,
# country effects, field, spatial+, biome-pooled SVC for tmax, and the space-time
# field on the central Sahel box (8-week steps, observed cells only).
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
python realdata/fit_glmm.py "$WORK/clean.parquet" --outcome "$OUT" --region africa_me \
    --models glm,country,field,spatialplus,svc,st --st-region central_sahel --st-step 8 \
    --out "$WORK/glmm"
