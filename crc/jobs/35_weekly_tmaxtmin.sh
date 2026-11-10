#!/bin/bash
#$ -N cs_tweekly
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=32G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: weekly means of the CHIRTS-ERA5 daily caches for tmax and tmin only (ready before hi/wbgt) (all cells 60S-72N).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
python realdata/covariates/chirts_weekly.py weekly tmax tmin
echo "rc=$?"; du -sh /groups/scastruc/conflict_spatial_work/covariates/*
