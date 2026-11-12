#!/bin/bash
#$ -N cs_semi
#$ -q long
#$ -l mem_free=48G
#$ -pe smp 4
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-20
#$ -tc 4
# Semi-synthetic check of spatially varying escalation on the real West Africa design (authorized 2026-11-12).
# Truth = fitted biv3 (saved hyperparameters, posterior mode of all effects), outcomes regenerated week by week.
#   tasks 1-10 : constant kappa (null): how much spread in kappa(s) does the fitted gamma field invent?
#   tasks 11-20: planted gamma field (Matern 500 km, SD = |fitted kappa_U->C|): is it recovered?
# Refit: biv3 + gamma fields, only the 4 gamma hyperparameters free, on the case-control subsample (r = 0.25).
# 4 tasks at once x 4 cores = 16 cores. One JSON per replicate in results_real/semisynth/ (aggregates only).
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
i=$SGE_TASK_ID
if [ $i -le 10 ]; then g=0; seed=$i; tag=null; else g=-1; seed=$((i - 10)); tag=planted; fi
python realdata/fit_bivariate.py "$CLEAN" --region west_sahel --land "$COUNTRIES" --st-step 13 --st-res 1.0 \
    --fixed-theta results_real/bivariate/west_sahel/summary_biv3.json --zero-frac 0.25 \
    --semisynth --sim-seed $seed --sim-gamma-sd $g --sim-out results_real/semisynth/west_sahel_${tag}_${seed}.json \
    --out "$WORK/semisynth" || exit 1
