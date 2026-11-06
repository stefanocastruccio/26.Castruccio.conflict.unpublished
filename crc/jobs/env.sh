# Shared settings for the CRC jobs; edit to match the CRC setup.
export DATA=/groups/scastruc/lmenical/drew/data
export WORK=/groups/scastruc/conflict_spatial_work
export REPO=${REPO:-/users/scastruc/projects/26.Castruccio.conflict.unpublished}
mkdir -p "$WORK/xgb" "$WORK/spatial"
# conda environment with pandas, xgboost, pyarrow, pymetis, geopandas
# CRC: dedicated env (the geostatistics env "geo" is left untouched)
module load conda >/dev/null 2>&1 || true
source activate /users/scastruc/envs/conflict 2>/dev/null || conda activate /users/scastruc/envs/conflict
export OMP_NUM_THREADS=${NSLOTS:-4} MKL_NUM_THREADS=${NSLOTS:-4} OPENBLAS_NUM_THREADS=${NSLOTS:-4}
# Required input (Task 1): Natural Earth 1:10m admin-0 countries, fetched to $WORK/inputs
export COUNTRIES=${COUNTRIES:-$WORK/inputs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp}
echo "host $(hostname) job ${JOB_ID}.${SGE_TASK_ID:-0} start $(date -u +%FT%TZ)"
