#!/bin/bash
#$ -N cs_psi
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-3
# Task 44: does escalation depend on conditions? kappa(s,t) = kappa + psi' z_st (Drew's hypotheses H1, H2a, H2b and the
# reporting check), in both directions, at the saved biv3 hyperparameters (--fixed-theta): one Laplace mode on all training
# rows and the fixed-effect intervals (--psi-only), no hyperparameter optimisation, no predictions.
# Needs Task 43 (V-Dem v2svstterr and v2stfisccap in the global table). Global table restricted to 2018-2024, the years of the
# saved fits. Task 1: ame, governance = v2svstterr; task 2: west_sahel, v2svstterr; task 3: ame, v2stfisccap (sensitivity).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
REG=(x ame west_sahel ame); STEP=(x 26 13 26); ZF=(x "--zero-frac 0.25" "" "--zero-frac 0.25")
GOV=(x v2svstterr v2svstterr v2stfisccap)
reg=${REG[$SGE_TASK_ID]}; gov=${GOV[$SGE_TASK_ID]}
R=results_real/bivariate/$reg; O="$WORK/bivariate/psi_${reg}_$gov"; mkdir -p "$O"
python realdata/fit_bivariate.py "$WORK/table_global/clean_global" --years 2018 2024 --region $reg --land "$COUNTRIES" \
    --st-step ${STEP[$SGE_TASK_ID]} --st-res 1.0 --models biv3 --fixed-theta "$R/summary_biv3.json" ${ZF[$SGE_TASK_ID]} \
    --psi gov,gov2,drought,heat,drought_ipc,remote --psi-gov $gov --psi-only \
    --summary-name psi_$gov.json --out "$O" || exit 1
cp "$O/psi_$gov.json" "$R/psi_$gov.json"
