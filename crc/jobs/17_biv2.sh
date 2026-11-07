#!/bin/bash
#$ -N cs_biv2
#$ -q long
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Joint model with an additional UNREST-SPECIFIC space-time field (biv2; review round 3).
#   task 1: West Africa and Sahel (priority);  task 2: Africa + ME (run after the current biv job)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel africa_me); STEP=(13 26)
i=$((SGE_TASK_ID - 1))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
    --st-step "${STEP[$i]}" --st-res 1.0 --models biv2 --summary-name summary_biv2.json \
    --out "$WORK/bivariate/${REG[$i]}"
