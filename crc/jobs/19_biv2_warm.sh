#!/bin/bash
#$ -N cs_biv2w
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 32
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# biv2 (shared + unrest-specific space-time fields), warm-started from the fitted biv
# hyperparameters of the same region, on the 64-core EPYC nodes, with a looser outer
# tolerance (CRC speed-up proposals 1-3, 2026-11-07).
#   task 1: West Africa and Sahel;  task 2: Africa + ME (replaces 1705023)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel africa_me); STEP=(13 26)
i=$((SGE_TASK_ID - 1))
export FIT_JOBS=16 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=60
python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
    --st-step "${STEP[$i]}" --st-res 1.0 --models biv2 --summary-name summary_biv2.json \
    --init-theta "results_real/bivariate/${REG[$i]}/summary_biv.json" \
    --out "$WORK/bivariate_warm/${REG[$i]}"
