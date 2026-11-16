#!/bin/bash
#$ -N cs_toys
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-30
# Task 48: the local toy checks on the CRC, one replicate per array task (synthetic data only).
#   tasks 1-20: S9 (three outcomes: repression, armed violence, unrest; psi per type vs the merged two-outcome model)
#   tasks 21-30: S10 (country effects on escalation: level u and drought sensitivity, BYM2, boundary tau test, per-country
#                classification against the planted truth)
# Merge afterwards (held job, crc/jobs/48_toys_merge.sh).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export OMP_NUM_THREADS=4 FIT_JOBS=4 N_JOBS=1
if [ "$SGE_TASK_ID" -le 20 ]; then
  N_REPS=10 SHARD="$((SGE_TASK_ID - 1))/20" python s9_split.py
else
  N_REPS=5 SHARD="$((SGE_TASK_ID - 21))/10" python s10_slopes.py
fi
echo "rc=$?"
