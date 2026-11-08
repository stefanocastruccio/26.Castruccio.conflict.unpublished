#!/bin/bash
#$ -N cs_select
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Region selection for the joint model WITHOUT the test period: condition the Africa + ME
# models on 2018-2021 (hyperparameters fixed at the 2018-2022 fit), score 2022, and pick the
# 10x10 degree window with the largest total gain in joint log score of biv over the product
# of the univariate static models (rule in realdata/joint_gain_map.py). 2023-2024 never enter.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT="$WORK/bivariate_val/africa_me"; mkdir -p "$OUT" results_real/bivariate/selection
python realdata/fit_bivariate.py "$CLEAN" --region africa_me --land "$COUNTRIES" \
    --st-step 26 --st-res 1.0 --validation --models uni_static,uni_static_u,biv \
    --fixed-theta results_real/bivariate/africa_me/summary_biv.json \
    --summary-name summary_val.json --out "$OUT"
python realdata/joint_gain_map.py "$OUT/test_pred_val.parquet" results_real/bivariate/selection/val2022
cp "$OUT/summary_val.json" results_real/bivariate/selection/
