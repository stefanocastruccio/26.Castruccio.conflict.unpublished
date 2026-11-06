#!/bin/bash
#$ -N cs_sims_merge
#$ -q long
#$ -l mem_free=16G
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Merge the shards of 08_sims.sh (each script reads all shard checkpoints, computes
# anything missing, and writes the summaries), then rebuild the paper tables.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO/sims"
export N_JOBS=${NSLOTS:-4} ST_WEEKLY=1
for S in s0_engine.py s1_residual_field.py s2_confounding.py s3_persistence.py s4_svc.py barrier_check.py; do
    python "$S" || echo "FAILED $S"
done
cd "$REPO" && python paper/make_tables.py
