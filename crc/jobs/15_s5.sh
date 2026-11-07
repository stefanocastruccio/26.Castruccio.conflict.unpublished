#!/bin/bash
#$ -N cs_s5
#$ -q long
#$ -l mem_free=16G
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-20
# S5 (bivariate: escalation vs shared episodes), 2 scenarios x 20 replicates sharded over 20 tasks.
# Afterwards run the merge once without SHARD:  cd sims && N_JOBS=1 python s5_bivariate.py
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
SHARD="$((SGE_TASK_ID - 1))/20" N_JOBS=1 python s5_bivariate.py
