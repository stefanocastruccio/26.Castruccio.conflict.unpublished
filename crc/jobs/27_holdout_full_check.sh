#!/bin/bash
#$ -N cs_hofull
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Check of the --zero-frac 0.25 approximation in 25_holdout_refit: africa_me, target conflict, fold 0,
# refitted on the full data (--zero-frac 1.0, all rows). Same arguments as 1712445.11 otherwise.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
python realdata/holdout_check.py "$CLEAN" --region africa_me --land "$COUNTRIES" --st-step 26 --st-res 1.0 \
    --fixed-theta "results_real/bivariate/africa_me/summary_jointpred.json" \
    --fixed-theta2 "results_real/bivariate/africa_me/summary_biv2.json" \
    --target conflict --refit --zero-frac 1.0 \
    --folds 5 --seed 0 --only-fold 0 --out "$WORK/holdout_refit_full/africa_me_conflict/fold0"
