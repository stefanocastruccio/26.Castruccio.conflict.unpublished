#!/bin/bash
#$ -N cs_e5merge
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=48G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: merge ERA5 into ppt (cells without CHIRPS; May-October level factor per month and 30-degree sector north of 50N,
# raw ERA5 otherwise) for weekly ppt, monthly totals and the daily cache; then recompute SPI-3 from the merged monthly totals.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
python realdata/covariates/era5_precip.py merge || exit 1
python realdata/covariates/spi3.py
echo "rc=$?"
