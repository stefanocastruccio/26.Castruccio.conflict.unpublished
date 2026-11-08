#!/bin/bash
#$ -N cs_zf
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -l mem_free=200G
#$ -pe smp 16
#$ -l h_rt=48:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
#$ -t 1-4
# Speed-up test (2026-11-08): hyperparameters on all events + a random fraction r of the
# non-events of each outcome (offset -log r, exact for the logistic likelihood), latent
# posterior on all training rows at those hyperparameters. Cold starts, default
# tolerances, so the timings compare with the full fits (biv: West Africa 16,252 s,
# AME 23,084 s). Compare hyperparameters, kappa and test scores with the full summaries.
#   1: west_sahel biv r=0.10   2: west_sahel biv r=0.25
#   3: africa_me  biv r=0.10   4: africa_me  biv2 r=0.10 (cold; the full biv2 needed a warm start)
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
REG=(west_sahel west_sahel africa_me africa_me); STEP=(13 13 26 26)
MOD=(biv biv biv biv2); R=(0.10 0.25 0.10 0.10)
i=$((SGE_TASK_ID - 1))
export FIT_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python realdata/fit_bivariate.py "$CLEAN" --region "${REG[$i]}" --land "$COUNTRIES" \
    --st-step "${STEP[$i]}" --st-res 1.0 --models "${MOD[$i]}" --summary-name "summary_${MOD[$i]}.json" \
    --zero-frac "${R[$i]}" --out "$WORK/bivariate_zf/${REG[$i]}_${MOD[$i]}_r${R[$i]}"
