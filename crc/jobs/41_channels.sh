#!/bin/bash
#$ -N cs_chan
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=120:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Do shared episodes (channel 3, the loading lambda) matter once shared persistent places (channel 2) are in?
#   step 1: 95% intervals for the saved biv3 (channels 1+2+3; shared static field = channel 2), no refit
#   step 2: biv3_nolam = biv3 with lambda = 0 (separate conflict and unrest space-time fields, shared static
#           field kept), warm-started from biv3, with --hessian intervals
# Task 1: ame (zero-frac 0.25, 26-week steps); task 2: west_sahel (all rows, 13-week steps).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
REG=(x ame west_sahel); STEP=(x 26 13); ZF=(x "--zero-frac 0.25" "")
reg=${REG[$SGE_TASK_ID]}; R=results_real/bivariate/$reg; O="$WORK/bivariate/chan_$reg"; mkdir -p "$O"
python realdata/fit_bivariate.py "$CLEAN" --region $reg --land "$COUNTRIES" --st-step ${STEP[$SGE_TASK_ID]} --st-res 1.0 \
    --models biv3 --fixed-theta "$R/summary_biv3.json" --hessian-only --summary-name ci_biv3.json --out "$O" ${ZF[$SGE_TASK_ID]} || exit 1
cp "$O/ci_biv3.json" "$R/ci_biv3.json"
python realdata/fit_bivariate.py "$CLEAN" --region $reg --land "$COUNTRIES" --st-step ${STEP[$SGE_TASK_ID]} --st-res 1.0 \
    --models biv3_nolam --init-theta "$R/summary_biv3.json" --hessian --summary-name summary_biv3_nolam.json --out "$O" ${ZF[$SGE_TASK_ID]} || exit 1
cp "$O/summary_biv3_nolam.json" "$R/summary_biv3_nolam.json"
