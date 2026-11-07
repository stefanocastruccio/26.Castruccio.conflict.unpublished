#!/bin/bash
#$ -N cs_biv
#$ -q long
#$ -l mem_free=200G
#$ -pe smp 4
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Joint conflict-unrest model (realdata/fit_bivariate.py): escalation vs shared episodes.
#   task 1: West Africa and Sahel box, shared space-time field at 13-week steps (1 degree)
#   task 2: Africa + Middle East, 26-week steps (1 degree); the heavier fit
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel africa_me); STEP=(13 26)
i=$((SGE_TASK_ID - 1))
# parallel finite-difference gradients (one forked process per hyperparameter), one BLAS
# thread each to avoid oversubscription
export FIT_JOBS=${NSLOTS:-4} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
    --st-step "${STEP[$i]}" --st-res 1.0 --out "$WORK/bivariate/${REG[$i]}"
