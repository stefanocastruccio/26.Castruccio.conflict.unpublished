#!/bin/bash
#$ -N cs_monfill
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=64G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: CHIRTS-ERA5 monthly Tmax/Tmin for the 540 cells 60S-40S added after their daily years were processed (owner: monthly fill).
# Downloads the two monthly files (~7.2 GB each), extracts those cells, deletes them.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=2
python realdata/covariates/chirts_weekly.py monthly_fill
echo "rc=$?"
