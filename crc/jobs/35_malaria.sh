#!/bin/bash
#$ -N cs_malaria
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 4
#$ -l mem_free=64G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35, item 6: MAP PfPR 2-10 2000-2025, population-weighted (GHSL) mean per 0.5-deg cell-year.
# Raw rasters first: bash realdata/covariates/map_download.sh (login node)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4
python realdata/covariates/malaria.py && python realdata/covariates/restrict_outputs.py
echo "rc=$?"
