#!/bin/bash
#$ -N cs_ce_merge
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=4G
#$ -pe smp 1
#$ -l h_rt=1:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 49 merge: boundary LR tests of the country effects.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
for reg in ame west_sahel; do
  python realdata/merge_tau.py "$WORK/bivariate/ce_$reg" results_real/bivariate/$reg/ce_tau_test.json
done
echo "rc=$?"
