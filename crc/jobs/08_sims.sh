#!/bin/bash
#$ -N cs_sims
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-61
# Simulation studies S0-S4 (sims/), split into shards. No data needed.
# Task ids:
#   1-4    S0
#   5-9    S1, default truth
#   10-24  S2 (checkpoint s2v2: adds the XGBoost fit to all training rows, est_full)
#   25-34  S3 base part (weekly space-time fits on); the three original truths are read
#          from their checkpoints, only the two slow truths (phi = 0.97) are new
#   35-40  S4, symmetric gamma (checkpoint s4v2: adds est_full)
#   41-50  S3 8-week space-time fit (Delta = 8) for all five truths   [new]
#   51-55  S1, truth calibrated to the real-data fit (S1_CONFIG=calibrated) [new]
#   56-61  S4, asymmetric gamma (S4_GAMMA=asym)                        [new]
# Round 3 rerun: submit with -t 10-61. Tasks 1-9 (S0, S1 default) are unchanged.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export N_JOBS=${NSLOTS:-4}
t=$SGE_TASK_ID
if   [ $t -le 4 ];  then S=s0_engine.py;        I=$((t-1));  N=4
elif [ $t -le 9 ];  then S=s1_residual_field.py; I=$((t-5));  N=5
elif [ $t -le 24 ]; then S=s2_confounding.py;    I=$((t-10)); N=15
elif [ $t -le 34 ]; then S=s3_persistence.py;    I=$((t-25)); N=10; export ST_WEEKLY=1 S3_PART=base
elif [ $t -le 40 ]; then S=s4_svc.py;            I=$((t-35)); N=6
elif [ $t -le 50 ]; then S=s3_persistence.py;    I=$((t-41)); N=10; export S3_PART=d8
elif [ $t -le 55 ]; then S=s1_residual_field.py; I=$((t-51)); N=5;  export S1_CONFIG=calibrated
else                     S=s4_svc.py;            I=$((t-56)); N=6;  export S4_GAMMA=asym
fi
SHARD="$I/$N" python "$S"
