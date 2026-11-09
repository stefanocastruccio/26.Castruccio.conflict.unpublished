#!/bin/bash
#$ -N cs_fmap
#$ -q long
#$ -l mem_free=64G
#$ -pe smp 4
#$ -l h_rt=04:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Application figure (Section 5): fitted intensity of conflict and unrest and the shared and
# unrest-only episode fields of the extended joint model (biv2), region ame, for the 26-week step
# 7 (late June to late December 2021: Afghanistan, Ethiopia, Sudan, the Sahel).
# Needs 31_fields.sh task 2 (it writes fitted_biv2.parquet; pull >= the commit adding export_fitted
# BEFORE 31.2 starts). Submit held on 31.2. Commit paper/figures/real_episodes.png.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
O="$WORK/bivariate_fields/ame"
python paper/make_fitted_map.py "$CLEAN" "$O/fitted_biv2.parquet" "$O/fields_biv2_st.csv.gz" 7 \
    paper/figures/real_episodes.png
