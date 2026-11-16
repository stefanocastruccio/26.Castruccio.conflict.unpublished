#!/bin/bash
#$ -N cs_ce2s
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Task 50: country effects on escalation, TWO-STAGE (replaces Task 49), 2018-2024, both directions:
#   kappa(c,s,t) = kappa + psi' z_st + u_c + sum_m v_m,c z_m,st
# Stage 1: the saved biv3 solved once at its hyperparameters; its latent effects and non-escalation fixed effects become a
# per-row offset. Stage 2: kappa, psi and the country effects (u, v for drought, heat, drought_ipc, remote; BYM2 for u and
# drought) on rows with history of the other outcome; their SDs optimised, Hessian intervals, boundary tau tests (in-job),
# per-country classification. Task 1: ame; task 2: west_sahel.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
REG=(x ame west_sahel); STEP=(x 26 13); ZF=(x "--zero-frac 0.25" "")
reg=${REG[$SGE_TASK_ID]}; R=results_real/bivariate/$reg; O="$WORK/bivariate/ce2s_$reg"; mkdir -p "$O"
python realdata/fit_bivariate.py "$WORK/table_global/clean_global" --years 2018 2024 --region $reg --land "$COUNTRIES" \
    --st-step ${STEP[$SGE_TASK_ID]} --st-res 1.0 --models biv3_cs --fixed-theta "$R/summary_biv3.json" ${ZF[$SGE_TASK_ID]} \
    --psi gov,gov2,drought,heat,drought_ipc,remote --psi-gov v2svstterr \
    --ce-terms u,drought,heat,drought_ipc,remote --ce-bym2 u,drought --two-stage --tau-test \
    --summary-name ce2s.json --out "$O" || exit 1
cp "$O/ce2s.json" "$R/ce2s_v2svstterr.json"
