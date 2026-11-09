#!/bin/bash
#$ -N cs_chirps
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 8
#$ -l mem_free=48G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35, item 1: CHIRPS v2.0 daily p25 -> weekly precipitation totals and monthly totals on the 0.5-deg cells, 1990-2025.
# Raw files are downloaded first on the login node: python realdata/covariates/chirps_weekly.py download
# Optional: YEARS="1990 1991" to process a subset.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python realdata/covariates/chirps_weekly.py process ${YEARS:+--years $YEARS}
echo "rc=$?"; du -sh /groups/scastruc/conflict_spatial_work/covariates/*
