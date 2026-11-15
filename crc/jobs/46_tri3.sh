#!/bin/bash
#$ -N cs_tri3
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 46: three-outcome model (armed violence, state repression, unrest) on the South Asia box (60-100E, 0-40N, includes Myanmar) 2018-2024 (PI's choice, 2026-11-15: repression is
# dense there, 23-52% of violence against civilians, against 10% in West Africa), with psi per type of
# conflict (governance = V-Dem v2svstterr), full empirical-Bayes fit on the case-control subsample (r = 0.25), then the
# Hessian intervals of the hyperparameters. Needs Task 45 ($WORK/table_global/actors).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
R=results_real/bivariate/south_asia; O="$WORK/bivariate/tri3_south_asia"; mkdir -p "$R"; mkdir -p "$O"
python realdata/fit_bivariate.py "$WORK/table_global/clean_global" --years 2018 2024 --region south_asia --land "$COUNTRIES" \
    --st-step 13 --st-res 1.0 --models tri3 --actors "$WORK/table_global/actors" --zero-frac 0.25 \
    --psi gov,gov2,drought,heat,drought_ipc,remote --psi-gov v2svstterr --hessian \
    --summary-name summary_tri3.json --out "$O" || exit 1
cp "$O/summary_tri3.json" "$R/summary_tri3.json"
