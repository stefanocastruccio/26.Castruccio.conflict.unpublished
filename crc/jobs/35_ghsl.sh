#!/bin/bash
#$ -N cs_ghsl
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 4
#$ -l mem_free=64G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35, item 3: GHSL GHS-POP epochs 1990-2025 (30 arcsec) summed to the 0.5-deg cells; yearly interpolation.
# Raw zips are downloaded first on the login node: python realdata/covariates/ghsl_pop.py download
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4
python realdata/covariates/ghsl_pop.py process
echo "rc=$?"
