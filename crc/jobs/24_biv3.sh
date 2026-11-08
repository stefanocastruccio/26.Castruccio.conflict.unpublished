#!/bin/bash
#$ -N cs_biv3
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# biv3 = biv2 + a shared STATIC field (CopyBlock of a barrier Matern, own loading), 2026-11-08.
# Motivation (dependence check, 1711662): the joint models under-predict co-occurrence in
# 2023-24 because the persistent hotspot signal moved from the static fields into the
# shared AR(1) space-time field, whose forecasts decay. A shared static field carries the
# persistent shared part without decay. Warm start from the fitted biv2; full data (no
# --zero-frac), so it compares directly with biv2. The summary has *_calibration blocks
# (observed / predicted, overall and in the top 1% / 10% of predicted co-occurrence).
#   task 1: West Africa and Sahel;  task 2: Africa + ME
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel africa_me); STEP=(13 26)
i=$((SGE_TASK_ID - 1))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
    --st-step "${STEP[$i]}" --st-res 1.0 --models uni_static,uni_static_u,biv3 --summary-name summary_biv3.json \
    --fixed-theta "results_real/bivariate/${REG[$i]}/summary_biv.json" \
    --init-theta "results_real/bivariate/${REG[$i]}/summary_biv2.json" \
    --out "$WORK/bivariate_biv3/${REG[$i]}"
