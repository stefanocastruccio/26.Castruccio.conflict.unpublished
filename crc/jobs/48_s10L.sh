#!/bin/bash
#$ -N cs_s10L
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=64G
#$ -pe smp 16
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-10
# Task 48b: S10 large: 48 countries of 5 x 5 cells (1,200 cells), 208 weeks: power of the per-country classification.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export OMP_NUM_THREADS=8 FIT_JOBS=8 N_JOBS=1
S10_SIZE=large N_REPS=5 SHARD="$((SGE_TASK_ID - 1))/10" python s10_slopes.py
echo "rc=$?"
