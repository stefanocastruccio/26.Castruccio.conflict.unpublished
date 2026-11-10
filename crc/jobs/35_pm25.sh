#!/bin/bash
#$ -N cs_pm25
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 1
#$ -l mem_free=64G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: ACAG PM2.5 monthly 0.1 deg -> 0.5-deg cells, monthly and weekly.
# Raw files first: python realdata/covariates/pm25.py download (login node)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=1
python realdata/covariates/pm25.py process
echo "rc=$?"
