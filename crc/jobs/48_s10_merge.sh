#!/bin/bash
#$ -N cs_s10_merge
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=8G
#$ -pe smp 1
#$ -l h_rt=2:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 48b merge: results/s10_raw.csv (small) and results/s10L_raw.csv (large); no refitting if all shards finished.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export OMP_NUM_THREADS=1 N_JOBS=1
S10_SIZE=small N_REPS=5 python s10_slopes.py
S10_SIZE=large N_REPS=5 python s10_slopes.py
echo "rc=$?"
