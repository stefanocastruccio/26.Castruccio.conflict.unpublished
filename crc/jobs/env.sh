# Shared settings for the CRC jobs; edit to match the CRC setup.
export DATA=/groups/scastruc/lmenical/drew/data
export WORK=/groups/scastruc/conflict_spatial_work
export REPO=${REPO:-$HOME/26.Castruccio.conflict.unpublished}
mkdir -p "$WORK/xgb" "$WORK/spatial"
# conda environment with pandas, xgboost, pyarrow, pymetis, geopandas
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate /users/scastruc/envs/geo
export OMP_NUM_THREADS=4
