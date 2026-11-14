#!/bin/bash
#$ -N cs_e5chk
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=04:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: preliminary consistency check of the ERA5 precipitation downloads on whole years (YEARS="2000 2015").
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
python realdata/covariates/era5_check.py ${YEARS:-2000 2015}
echo "rc=$?"
