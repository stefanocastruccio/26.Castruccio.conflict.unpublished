#!/bin/bash
#$ -N cs_s10
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-10
# Task 48b: S10 (country effects on escalation) after the fixes, small size: 16 countries, 104 weeks. One replicate per task.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export OMP_NUM_THREADS=4 FIT_JOBS=4 N_JOBS=1
S10_SIZE=small N_REPS=5 SHARD="$((SGE_TASK_ID - 1))/10" python s10_slopes.py
echo "rc=$?"
