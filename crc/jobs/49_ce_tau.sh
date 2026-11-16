#!/bin/bash
#$ -N cs_ce_tau
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-20
# Task 49, tau test: one reduced fit per task (the model of 49_ce.sh without one country effect); tasks 1-10 ame, 11-20
# west_sahel; within a region: conflict u, drought, heat, drought_ipc, remote, then unrest (the same five).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
i=$((SGE_TASK_ID - 1)); r=$((i / 10)); k=$((i % 10))
REG=(ame west_sahel); STEP=(26 13); ZF=("--zero-frac 0.25" "")
TERMS=(conflict:u conflict:drought conflict:heat conflict:drought_ipc conflict:remote unrest:u unrest:drought unrest:heat unrest:drought_ipc unrest:remote)
reg=${REG[$r]}; drop=${TERMS[$k]}; R=results_real/bivariate/$reg; O="$WORK/bivariate/ce_$reg"; mkdir -p "$O"
python realdata/fit_bivariate.py "$WORK/table_global/clean_global" --years 2018 2024 --region $reg --land "$COUNTRIES" \
    --st-step ${STEP[$r]} --st-res 1.0 --models biv3_cs --fixed-theta "$R/summary_biv3.json" ${ZF[$r]} \
    --psi gov,gov2,drought,heat,drought_ipc,remote --psi-gov v2svstterr \
    --ce-terms u,drought,heat,drought_ipc,remote --ce-bym2 u,drought --tau-drop $drop \
    --summary-name "ce_tau_${drop/:/_}.json" --out "$O" || exit 1
