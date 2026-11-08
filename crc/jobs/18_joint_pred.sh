#!/bin/bash
#$ -N cs_jpred
#$ -q long
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Joint prediction: does the joint model predict conflict and unrest TOGETHER better than
# two univariate models? Joint log score of (conflict, unrest) per cell-week, the joint model
# with and without its dependence, conflict given same-week unrest, co-occurrence.
# Reuses the hyperparameters of the earlier fits (summary_biv.json: uni_static, uni_st, biv), so
# only the univariate UNREST models are optimised; the rest is one posterior mode + sampling.
#   task 1: West Africa and Sahel;  task 2: Africa + ME
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel africa_me); STEP=(13 26)
i=$((SGE_TASK_ID - 1))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
    --st-step "${STEP[$i]}" --st-res 1.0 --models uni_static,uni_st,uni_static_u,uni_st_u,biv \
    --fixed-theta "$WORK/bivariate/${REG[$i]}/summary_biv.json" \
    --summary-name summary_jointpred.json --out "$WORK/bivariate/${REG[$i]}"
