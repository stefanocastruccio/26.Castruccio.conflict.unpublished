#!/bin/bash
#$ -N cs_chirts
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=16G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1990-2026
#$ -tc 4
# Task 35, item 2: CHIRTS-ERA5 daily Tmax/Tmin, one task per year (download -> daily 0.5-deg cell means -> raw deleted).
# At most 4 tasks at once (8 cores, about 60 GB raw in flight). 2026 is fetched for the last week of 2025.
# Afterwards: python realdata/covariates/chirts_weekly.py weekly  (or 35_chirts_weekly via qsub -hold_jid)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python realdata/covariates/chirts_weekly.py daily $SGE_TASK_ID
echo "rc=$?"
