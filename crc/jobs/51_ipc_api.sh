#!/bin/bash
#$ -N cs_ipc51
#$ -q long@@crc_d32cepyc,long@@acms_d32cepyc
#$ -pe smp 2
#$ -l mem_free=96G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -j y
#$ -o /groups/scastruc/conflict_spatial_work/logs/
# Task 51: IPC 2017-2025 from the IPC API (raw responses already downloaded on the login node into raw/ipc_api/, so no key or
# network is needed here), then the IPC columns of the global table: join_cov --var ipc, assemble.
source /users/scastruc/projects/26.Castruccio.conflict.unpublished/crc/jobs/env.sh
cd "$REPO"
python realdata/covariates/ipc.py --source api || exit 1
python realdata/table/join_cov.py --var ipc || exit 1
python realdata/table/assemble.py || exit 1
python realdata/covariates/ipc_coverage.py || exit 1
echo "rc=$?"
