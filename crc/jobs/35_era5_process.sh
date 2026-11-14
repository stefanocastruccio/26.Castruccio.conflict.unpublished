#!/bin/bash
#$ -N cs_e5proc
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 35: ERA5 daily per cell (daily/ppt_era5_<y>.npz) for the given year ranges (RANGES="1990 2006 2012 2019"); with FACTORS=1
# also print the CHIRPS/ERA5 scaling factors from the processed years (grid/ppt_era5_scaling.csv). Skips years already processed.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
set -- ${RANGES:-1990 2026}
while [ $# -ge 2 ]; do python realdata/covariates/era5_precip.py process --years $1 $2 || exit 1; shift 2; done
[ "${FACTORS:-0}" = 1 ] && python -c "import sys; sys.path.insert(0,'realdata/covariates'); import era5_precip as E, common as C; print(E.scaling_factors(C.cells()).round(3))"
echo "rc=$?"
