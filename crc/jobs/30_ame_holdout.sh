#!/bin/bash
#$ -N cs_ameho
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=72:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-10
# Country hold-out with per-fold refits (as 25_holdout_refit.sh) for region "ame" (no Europe).
# Submit held on 29_ame_fits.sh (needs results_real/bivariate/ame/summary_{jointpred,biv2}.json).
#   tasks 1-5 target conflict folds 0-4;  tasks 6-10 target unrest folds 0-4
# Pool afterwards:  python realdata/holdout_check.py --merge results_real/bivariate/holdout_refit/ame_<target>.json \
#                       $WORK/holdout_refit/ame_<target>/fold*
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
i=$((SGE_TASK_ID - 1)); TGS=(conflict unrest); TG=${TGS[$((i / 5))]}; K=$((i % 5))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
python realdata/holdout_check.py "$CLEAN" --region ame --land "$COUNTRIES" --st-step 26 --st-res 1.0 \
    --fixed-theta results_real/bivariate/ame/summary_jointpred.json \
    --fixed-theta2 results_real/bivariate/ame/summary_biv2.json \
    --target "$TG" --refit --zero-frac 0.25 \
    --folds 5 --seed 0 --only-fold "$K" --out "$WORK/holdout_refit/ame_${TG}/fold$K"
