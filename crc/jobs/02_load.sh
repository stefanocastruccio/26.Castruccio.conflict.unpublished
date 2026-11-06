#!/bin/bash
#$ -N cs_load
#$ -q largemem
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Build the repaired (clean) and the original (duplicated) tables from the 50 partitions.
# Both inputs are required (crc/TASKS.md, Task 1):
#   COUNTRIES  Natural Earth 1:10m admin-0 countries shapefile (set in env.sh)
#   GDP_CSV    World Bank GDP, current US$ (NY.GDP.MKTP.CD), CSV as downloaded
# SGE runs a spooled copy of the script, so $(dirname "$0") is not the repo
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
GDP_CSV=${GDP_CSV:-$WORK/inputs/GDP.csv}
for f in "$COUNTRIES" "$GDP_CSV"; do
    [ -f "$f" ] || { echo "missing required input: $f"; exit 1; }
done
python realdata/load_partitions.py "$DATA/partitions" "$WORK/clean.parquet" \
    --countries "$COUNTRIES" --gdp-csv "$GDP_CSV"
python realdata/load_partitions.py "$DATA/partitions" "$WORK/orig.parquet" \
    --countries "$COUNTRIES" --keep-duplicates
