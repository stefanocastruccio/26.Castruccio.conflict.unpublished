#!/bin/bash
#$ -N cs_dep
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 4
#$ -l mem_free=32G
#$ -l h_rt=02:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Dependence diagnostic on the existing 2023-24 joint-prediction parquets (no refitting):
# observed vs predicted co-occurrence, and the odds ratio of y_conflict on same-week / last-week
# unrest given logit(p_conflict) of each model, overall and in the selected windows, Yemen and Sudan.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python realdata/dependence_check.py "$WORK" "$WORK/bivariate_val/africa_me/test_pred_val.parquet" \
    results_real/bivariate/selection/val2022_windows.csv results_real/bivariate/selection/dependence_check.json
echo "rc=$?"
