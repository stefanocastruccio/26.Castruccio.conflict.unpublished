#!/bin/bash
#$ -N cs_dyncmp
#$ -q long
#$ -l mem_free=60G
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Like-for-like comparisons for fit_dynamic.py (task 1 = conflict, 2 = unrest):
#   compare_<outcome>_st.json       2023-2024: static hybrid vs space-time hybrids vs boosting
#   compare_<outcome>_rolling.json  2024 only: static fits (trained to 2022) vs rolling fits
#                                   (trained to 2023); compare.py keeps the common rows
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}; OUTC=${OUT^}
S=$WORK/spatial/$OUT; XG=$WORK/xgb; D=$WORK/dynamic/$OUT
mkdir -p "$WORK/compare"
python realdata/compare.py \
    --pred xgb=$XG/pred_corrected_${OUTC}.parquet:logit \
    --pred xgb_te=$XG/pred_corrected_te_${OUTC}.parquet:logit \
    --pred hybrid_iid=$S/test_pred.parquet:p_iid \
    --pred hybrid_field=$S/test_pred.parquet:p_field \
    --pred hybrid_st13=$D/test_pred_st13_r1.parquet:p_st13_r1 \
    --pred hybrid_st26=$D/test_pred_st26_r1.parquet:p_st26_r1 \
    --lag $S/test_pred.parquet --ref xgb --out "$WORK/compare/compare_${OUT}_st.json"
python realdata/compare.py \
    --pred xgb=$XG/pred_corrected_${OUTC}.parquet:logit \
    --pred xgb_te=$XG/pred_corrected_te_${OUTC}.parquet:logit \
    --pred hybrid_field=$S/test_pred.parquet:p_field \
    --pred hybrid_iid=$S/test_pred.parquet:p_iid \
    --pred xgb_crossfit=$S/test_pred.parquet:p_xgb_crossfit \
    --pred hybrid_field_roll=$D/test_pred_2024.parquet:p_field_roll \
    --pred hybrid_iid_roll=$D/test_pred_2024.parquet:p_iid_roll \
    --pred xgb_crossfit_roll=$D/test_pred_2024.parquet:p_xgb_crossfit_roll \
    --lag $S/test_pred.parquet --ref xgb --out "$WORK/compare/compare_${OUT}_rolling.json"
