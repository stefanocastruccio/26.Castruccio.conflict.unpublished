#!/bin/bash
#$ -N cs_ho
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-10
# Country hold-out experiment (2026-11-08): conflict unobserved in one fold of countries
# (all weeks), unrest observed; models re-conditioned at their saved hyperparameters.
#   tasks 1-5: west_sahel folds 0-4;  tasks 6-10: africa_me folds 0-4
# Afterwards pool the folds of each region:
#   python realdata/holdout_check.py --merge results_real/bivariate/holdout/<region>.json \
#       $WORK/holdout/<region>/fold*
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
i=$((SGE_TASK_ID - 1))
if [ "$i" -lt 5 ]; then REG=west_sahel; STEP=13; K=$i; else REG=africa_me; STEP=26; K=$((i - 5)); fi
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python realdata/holdout_check.py "$CLEAN" --region "$REG" --land "$COUNTRIES" --st-step "$STEP" --st-res 1.0 \
    --fixed-theta "results_real/bivariate/$REG/summary_biv.json" \
    --fixed-theta2 "results_real/bivariate/$REG/summary_biv2.json" \
    --folds 5 --seed 0 --only-fold "$K" --out "$WORK/holdout/$REG/fold$K"
