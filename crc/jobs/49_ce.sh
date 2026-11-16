#!/bin/bash
#$ -N cs_ce
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Task 49: country effects on escalation (model biv3_cs) on real data, 2018-2024, both directions:
#   kappa(c,s,t) = kappa + psi' z_st + u_c + sum_m v_m,c z_m,st
# psi terms gov, gov2 (V-Dem territorial authority), drought, heat, drought x IPC 3+, remoteness (centred on rows with history of
# the other outcome); country effects u (level) and v for drought, heat, drought_ipc, remote; BYM2 (spatial) for u and drought,
# iid for the others; PC prior P(tau > 0.2) = 0.05. biv3 hyperparameters fixed at the saved fit; the country-effect SDs
# optimised, their Hessian intervals, per-country classification. Task 1: ame; task 2: west_sahel.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
REG=(x ame west_sahel); STEP=(x 26 13); ZF=(x "--zero-frac 0.25" "")
reg=${REG[$SGE_TASK_ID]}; R=results_real/bivariate/$reg; O="$WORK/bivariate/ce_$reg"; mkdir -p "$O"
python realdata/fit_bivariate.py "$WORK/table_global/clean_global" --years 2018 2024 --region $reg --land "$COUNTRIES" \
    --st-step ${STEP[$SGE_TASK_ID]} --st-res 1.0 --models biv3_cs --fixed-theta "$R/summary_biv3.json" ${ZF[$SGE_TASK_ID]} \
    --psi gov,gov2,drought,heat,drought_ipc,remote --psi-gov v2svstterr \
    --ce-terms u,drought,heat,drought_ipc,remote --ce-bym2 u,drought \
    --summary-name ce_main.json --out "$O" || exit 1
cp "$O/ce_main.json" "$R/ce_v2svstterr.json"
