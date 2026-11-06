#!/bin/bash
#$ -N cs_check
#$ -q long
#$ -l mem_free=120G
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Direct checks on the ORIGINAL partitions: which columns differ across copies, and
# leakage of the current week into the original 4-week counts.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
mkdir -p "$WORK/data"
python realdata/check_duplicates.py "$DATA/partitions" "$WORK/data/check_duplicates.json"
