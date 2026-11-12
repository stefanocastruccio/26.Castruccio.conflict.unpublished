#!/bin/bash
#$ -N cs_covv
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=48G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 40, part B for some variables: qsub -v VARS="tmax tmin" crc/jobs/40_cov_var.sh
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
for v in $VARS; do python realdata/table/join_cov.py --var $v || exit 1; done
echo "rc=$?"
