#!/bin/bash
#$ -N cs_load
#$ -q largemem
#$ -pe smp 4
#$ -cwd
#$ -j y
# Build the repaired (clean) and the original (duplicated) tables from the 50 partitions.
source "$(dirname "$0")/env.sh"
cd "$REPO"
python realdata/load_partitions.py "$DATA/partitions" "$WORK/clean.parquet" \
    --countries "$DATA/world_map/ne_10m_admin_0_countries.shp" \
    --gdp-csv "$DATA/socioeconomic/GDP.csv"
python realdata/load_partitions.py "$DATA/partitions" "$WORK/orig.parquet" \
    --countries "$DATA/world_map/ne_10m_admin_0_countries.shp" --keep-duplicates
