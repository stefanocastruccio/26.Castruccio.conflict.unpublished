#!/bin/bash
#$ -N cs_ucdp
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 4
#$ -l mem_free=64G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 47: UCDP GED on the ACLED table's cells and weeks ($WORK/table_global/ucdp/year=<y>.parquet). No analysis.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4
python realdata/table/ucdp.py "$WORK/ucdp/raw/$GED_CSV"
echo "rc=$?"
