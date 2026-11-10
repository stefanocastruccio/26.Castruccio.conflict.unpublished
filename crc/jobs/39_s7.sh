#!/bin/bash
#$ -N cs_s7
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-10
#$ -tc 4
# S7: recovery of region-specific loadings and of a spatially varying kappa(s), and no spurious
# variation when both are constant (sims/s7_varying.py). 2 scenarios x 20 replicates, sharded over 10
# tasks, at most 4 at once (16 cores). Afterwards merge once without SHARD: cd sims && N_JOBS=1 python s7_varying.py
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export FIT_JOBS=1 OMP_NUM_THREADS=4
SHARD="$((SGE_TASK_ID - 1))/10" N_JOBS=1 python s7_varying.py
