#!/bin/bash
#$ -N cs_cov
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=48G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 40, part B: covariate files ($WORK/table_global/cov/<var>/).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4
python realdata/table/join_cov.py --var all
echo "rc=$?"; du -sh /groups/scastruc/conflict_spatial_work/table_global
