#!/bin/bash
#$ -N cs_pptdaily
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 1
#$ -l mem_free=48G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: daily CHIRPS precipitation per 0.5-deg cell (daily/ppt_<year>.npz), kept before the raw files are deleted.
# Raw files are downloaded first on the login node: python realdata/covariates/chirps_weekly.py download
# Optional: YEARS="1990 1991" to process a subset.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=1
python realdata/covariates/chirps_weekly.py daily
echo "rc=$?"; du -sh /groups/scastruc/conflict_spatial_work/covariates/*
