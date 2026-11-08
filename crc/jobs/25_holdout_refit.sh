#!/bin/bash
#$ -N cs_hor
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=72:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-20
# Country hold-out with the hyperparameters RE-ESTIMATED in every fold (2026-11-08), so the
# held-out countries' target never enters any estimate; both directions:
#   target conflict (conflict hidden, unrest observed) and target unrest (the mirror).
# Every fit uses --zero-frac 0.25 (all events + 25% of non-events for the hyperparameters;
# reproduced the full fit in 1711777.2) and starts from the saved full-data hyperparameters.
#   tasks 1-5  west_sahel conflict folds 0-4     tasks 6-10  west_sahel unrest folds 0-4
#   tasks 11-15 africa_me conflict folds 0-4     tasks 16-20 africa_me unrest folds 0-4
# Pool afterwards, per region and target:
#   python realdata/holdout_check.py --merge results_real/bivariate/holdout_refit/<region>_<target>.json \
#       $WORK/holdout_refit/<region>_<target>/fold*
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
i=$((SGE_TASK_ID - 1))
REGS=(west_sahel africa_me); STEPS=(13 26); TGS=(conflict unrest)
REG=${REGS[$((i / 10))]}; STEP=${STEPS[$((i / 10))]}; TG=${TGS[$(((i / 5) % 2))]}; K=$((i % 5))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
python realdata/holdout_check.py "$CLEAN" --region "$REG" --land "$COUNTRIES" --st-step "$STEP" --st-res 1.0 \
    --fixed-theta "results_real/bivariate/$REG/summary_jointpred.json" \
    --fixed-theta2 "results_real/bivariate/$REG/summary_biv2.json" \
    --target "$TG" --refit --zero-frac 0.25 \
    --folds 5 --seed 0 --only-fold "$K" --out "$WORK/holdout_refit/${REG}_${TG}/fold$K"
