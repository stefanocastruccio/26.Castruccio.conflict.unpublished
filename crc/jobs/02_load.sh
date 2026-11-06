#!/bin/bash
#$ -N cs_load
#$ -q long
#$ -l mem_free=120G
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
# ACLED coverage: the official table if Task 1b produced it, otherwise the estimated rule
COV_ARG=()
[ -f "$WORK/inputs/acled_coverage.csv" ] && COV_ARG=(--coverage-csv "$WORK/inputs/acled_coverage.csv")
# CLEAN_OUT: output of the repaired table (default $CLEAN from env.sh); ONLY_CLEAN=1 skips orig.parquet
CLEAN_OUT=${CLEAN_OUT:-$CLEAN}
python realdata/load_partitions.py "$DATA/partitions" "$CLEAN_OUT" \
    --countries "$COUNTRIES" --gdp-csv "$GDP_CSV" "${COV_ARG[@]}"
[ "${ONLY_CLEAN:-0}" = 1 ] && exit 0
python realdata/load_partitions.py "$DATA/partitions" "$WORK/orig.parquet" \
    --countries "$COUNTRIES" --keep-duplicates
