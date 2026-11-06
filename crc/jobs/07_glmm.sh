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
# Parametric models (Africa + Middle East): lags, covariates (missingness indicators for
# the national indicators and IPC), national indicators, country effects, field + iid cell
# effect, spatial+, biome-pooled SVC for tmax, and the space-time comparison on the central
# Sahel box (w on the buffered lattice with sea barrier, v on the observed cells, 8-week steps).
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}
python realdata/fit_glmm.py "$CLEAN" --outcome "$OUT" --region africa_me \
    --models glm,country,field,spatialplus,svc,st --st-region central_sahel --st-step 8 --land "$COUNTRIES" \
    --out "$WORK/glmm"
