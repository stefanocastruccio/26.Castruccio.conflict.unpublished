#!/bin/bash
#$ -N cs_updcmp
#$ -q long
#$ -l mem_free=60G
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# 2023-2024, every model updated quarterly with the data before each quarter
# (the cross-fitted offset and all boosting models already use the past 4 weeks' events).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
OUTS=(conflict unrest); OUT=${OUTS[$((SGE_TASK_ID - 1))]}; OUTC=${OUT^}
S=$WORK/spatial/$OUT; XG=$WORK/xgb; D=$WORK/dynamic/$OUT
python realdata/compare.py \
    --pred xgb=$XG/pred_corrected_${OUTC}.parquet:logit \
    --pred xgb_te=$XG/pred_corrected_te_${OUTC}.parquet:logit \
    --pred xgb_te_upd=$XG/pred_corrected_te_${OUTC}.parquet:logit_upd \
    --pred hybrid_field=$S/test_pred.parquet:p_field \
    --pred hybrid_field_upd=$D/test_pred_update.parquet:p_field_upd \
    --pred hybrid_iid_upd=$D/test_pred_update.parquet:p_iid_upd \
    --pred hybrid_st_upd=$D/test_pred_update.parquet:p_st_upd \
    --lag $S/test_pred.parquet --ref xgb --out "$WORK/compare/compare_${OUT}_update.json"
