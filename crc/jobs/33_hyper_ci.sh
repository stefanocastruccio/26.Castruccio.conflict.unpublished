#!/bin/bash
#$ -N cs_ci
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Approximate 95% intervals for the hyperparameters of the joint (biv) and extended (biv2) models,
# at the SAVED estimates (--fixed-theta: no refit, no predictions). Intervals come from the Hessian
# of the Laplace log posterior of theta (central differences, 2 p^2 evaluations in parallel).
# AME on the same case-control subsample as its fit (r = 0.25, seed 0); West Africa on all rows.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() {  # region step model saved extra...
  local reg=$1 step=$2 model=$3 saved=$4; shift 4
  local R=results_real/bivariate/$reg O="$WORK/bivariate/ci_$reg"; mkdir -p "$O"
  python realdata/fit_bivariate.py "$CLEAN" --region "$reg" --land "$COUNTRIES" --st-step "$step" --st-res 1.0 \
      --models "$model" --fixed-theta "$R/$saved" --hessian-only --summary-name "ci_$model.json" --out "$O" "$@" || exit 1
  cp "$O/ci_$model.json" "$R/ci_$model.json"
}
run ame 26 biv summary_biv.json --zero-frac 0.25
run ame 26 biv2 summary_biv2.json --zero-frac 0.25
run west_sahel 13 biv summary_biv.json
run west_sahel 13 biv2 summary_biv2.json
