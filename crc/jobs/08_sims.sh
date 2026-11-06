#!/bin/bash
#$ -N cs_sims
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-40
# Simulation studies S0-S4 (sims/), split into shards. No data needed.
# Task ids: S0 1-4, S1 5-9, S2 10-24, S3 25-34 (with weekly space-time fits), S4 35-40.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export N_JOBS=${NSLOTS:-4}
t=$SGE_TASK_ID
if   [ $t -le 4 ];  then S=s0_engine.py;        I=$((t-1));  N=4
elif [ $t -le 9 ];  then S=s1_residual_field.py; I=$((t-5));  N=5
elif [ $t -le 24 ]; then S=s2_confounding.py;    I=$((t-10)); N=15
elif [ $t -le 34 ]; then S=s3_persistence.py;    I=$((t-25)); N=10; export ST_WEEKLY=1
else                     S=s4_svc.py;            I=$((t-35)); N=6
fi
SHARD="$I/$N" python "$S"
