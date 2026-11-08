#!/bin/bash
#$ -N cs_s5
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-40
# S5 v4 (bivariate: escalation vs shared episodes): 4 scenarios (incl. unrest-only episodes) x 20
# replicates, 5 estimators (incl. biv_st2), sharded over 40 tasks; checkpoint s5v4_ckpt.jsonl; joint, conditional and marginal prediction all scored by Monte Carlo over the same posterior draws (v4).
# Afterwards run the merge once without SHARD:  cd sims && N_JOBS=1 python s5_bivariate.py
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export FIT_JOBS=4 OMP_NUM_THREADS=1
SHARD="$((SGE_TASK_ID - 1))/40" N_JOBS=1 python s5_bivariate.py
