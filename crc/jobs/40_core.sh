#!/bin/bash
#$ -N cs_core
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 4
#$ -l mem_free=96G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 40, part A: global cell-week core table from the raw ACLED events ($WORK/table_global/core/year=<y>/part.parquet).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4
python realdata/table/core.py
echo "rc=$?"; du -sh /groups/scastruc/conflict_spatial_work/table_global
