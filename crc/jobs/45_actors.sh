#!/bin/bash
#$ -N cs_actors
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 4
#$ -l mem_free=96G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 45: conflict split by actor ($WORK/table_global/actors/year=<y>.parquet): Repression (violence against civilians by
# state forces) and Armed (battles, explosions/remote violence, violence against civilians by other actors), with histories.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4
python realdata/table/actors.py
echo "rc=$?"
