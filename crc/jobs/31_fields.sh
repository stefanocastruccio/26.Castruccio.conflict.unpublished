#!/bin/bash
#$ -N cs_fields
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-2
# Posterior means of the latent fields of the joint models (biv, biv2, biv3), for the maps of
# Section 5: each model is CONDITIONED on its saved hyperparameters (one Newton solve, no
# optimisation) and fit_bivariate.py writes fields_<model>_{static,st}.csv.gz (model output only).
#   task 1: west_sahel (now)     task 2: ame (submit held on 29_ame_fits.sh)
# Commit results_real/bivariate/<region>/fields_*.csv.gz.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel ame); STEP=(13 26); i=$((SGE_TASK_ID - 1))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=results_real/bivariate/${REG[$i]}; O="$WORK/bivariate_fields/${REG[$i]}"; mkdir -p "$O"
for M in biv biv2 biv3; do
  python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
      --st-step "${STEP[$i]}" --st-res 1.0 --models "$M" --fixed-theta "$R/summary_$M.json" \
      --summary-name "summary_${M}_fields.json" --out "$O" || exit 1
  cp "$O"/fields_${M}_*.csv.gz "$R/"
done
