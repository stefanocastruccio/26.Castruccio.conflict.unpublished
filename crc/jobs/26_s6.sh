#!/bin/bash
#$ -N cs_s6
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-40
# S6 (prediction where one outcome is unobserved): S5 generator, scenarios no_escalation,
# escalation, escalation_uST; 20 replicates; target conflict and target unrest; a random
# 6 x 6-cell block where the target is unobserved in all weeks; uni_field, uni_st, biv_st,
# biv_st2 refitted on what remains. Sharded over 40 tasks; checkpoint s6_ckpt.jsonl.
# Afterwards merge once without SHARD:  cd sims && N_JOBS=1 python s6_holdout.py
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export FIT_JOBS=4 OMP_NUM_THREADS=1
SHARD="$((SGE_TASK_ID - 1))/40" N_JOBS=1 python s6_holdout.py
