#!/bin/bash
#$ -N cs_vary
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=120:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Prototype of the global model on "ame" (Africa + Middle East, Europe excluded), with three regions
# by country (africa, middle_east, south_asia = AFG + PAK). Warm-started from the saved biv2.
# --zero-frac 0.25 as for biv2; --hessian adds 95% intervals for every hyperparameter.
#   biv2_rl  : one loading of the shared space-time field per region
#   biv2_rlf : + spatially varying cross effects kappa(s) = kappa_0 + gamma(s), Matern SVC, both directions
#   biv2_rlk : biv2_rl + region-specific kappa as fixed interactions (run alone: it adds covariates)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
R=results_real/bivariate/ame; O="$WORK/bivariate/ame"; mkdir -p "$R" "$O"
run() {  # models summary
  python realdata/fit_bivariate.py "$CLEAN" --region ame --land "$COUNTRIES" --st-step 26 --st-res 1.0 \
      --models "$1" --summary-name "$2" --init-theta "$R/summary_biv2.json" --zero-frac 0.25 --hessian --out "$O" || exit 1
  cp "$O/$2" "$R/$2"
}
run biv2_rl summary_biv2_rl.json
run biv2_rlf summary_biv2_rlf.json
run biv2_rlk summary_biv2_rlk.json
