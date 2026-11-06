#!/bin/bash
#$ -N cs_load
#$ -q largemem
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Build the repaired (clean) and the original (duplicated) tables from the 50 partitions.
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
python realdata/load_partitions.py "$DATA/partitions" "$WORK/clean.parquet" \
    --countries "$COUNTRIES" $( [ -f "$DATA/socioeconomic/GDP.csv" ] && echo --gdp-csv "$DATA/socioeconomic/GDP.csv" )
python realdata/load_partitions.py "$DATA/partitions" "$WORK/orig.parquet" \
    --countries "$COUNTRIES" --keep-duplicates
