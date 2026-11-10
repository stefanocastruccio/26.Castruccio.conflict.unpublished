#!/bin/bash
#$ -N cs_covglob
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=64G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: redo the outputs that had been restricted to the five regions, now for all cells 40S-72N (owner, 2026-11-09:
# global model first). Uses the raw files already on disk (no download).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=2
set -e
python realdata/covariates/chirps_weekly.py process
python realdata/covariates/spi3.py
python realdata/covariates/ghsl_pop.py process
python realdata/covariates/malaria.py
python realdata/covariates/pm25.py process
echo "rc=0"; du -sh /groups/scastruc/conflict_spatial_work/covariates/grid/*
