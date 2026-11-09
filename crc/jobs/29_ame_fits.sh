#!/bin/bash
#$ -N cs_ame
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Region "ame" (2026-11-09): the africa_me box WITHOUT its European countries (EUROPE in
# realdata/fit_bivariate.py; cells assigned by nearest Natural Earth polygon). Replaces africa_me
# in the paper. Hyperparameters from the case-control subsample (--zero-frac 0.25), warm starts:
#   A: uni_static, uni_st, uni_static_u, uni_st_u, biv  (biv starts from africa_me's biv)
#   B: biv2  (from A's biv)          C: biv3  (from B's biv2)
# Each step copies its summary into results_real/bivariate/ame/ (commit those three files).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LAPLACE_FTOL=1e-7 LAPLACE_GTOL=1e-3 LAPLACE_MAXITER=80
R=results_real/bivariate/ame; O="$WORK/bivariate/ame"; mkdir -p "$R" "$O"
run() {  # models summary init
  python realdata/fit_bivariate.py "$CLEAN" --region ame --land "$COUNTRIES" --st-step 26 --st-res 1.0 \
      --models "$1" --summary-name "$2" --init-theta "$3" --zero-frac 0.25 --out "$O" || exit 1
  cp "$O/$2" "$R/$2"
}
run uni_static,uni_st,uni_static_u,uni_st_u,biv summary_jointpred.json results_real/bivariate/africa_me/summary_biv.json
cp "$R/summary_jointpred.json" "$R/summary_biv.json"
run biv2 summary_biv2.json "$R/summary_biv.json"
run biv3 summary_biv3.json "$R/summary_biv2.json"
