#!/bin/bash
# Task 43: V-Dem Country-Year (Full+Others) v16.0 from the V-Dem Institute's R data package vdemdata (GitHub, "v16 release",
# commit f4dd269, 2026-03-17; v-dem.net itself needs a form). Writes $WORK/covariates/raw/vdem/vdem_v16_subset.csv with
# country_name, country_text_id, year (>= 1990), v2x_libdem, v2stfisccap, v2svstterr, read by national.py. Needs module R/4.5.2.
set -e
D=/groups/scastruc/conflict_spatial_work/covariates/raw/vdem; mkdir -p "$D"
curl -sSL -o "$D/vdem.RData" https://github.com/vdeminstitute/vdemdata/raw/f4dd269/data/vdem.RData
Rscript --vanilla -e 'e <- new.env(); load(file.path(commandArgs(TRUE)[1], "vdem.RData"), envir = e); v <- e$vdem
k <- c("country_name", "country_text_id", "year", "v2x_libdem", "v2stfisccap", "v2svstterr")
write.csv(v[v$year >= 1990, k], file.path(commandArgs(TRUE)[1], "vdem_v16_subset.csv"), row.names = FALSE)' "$D"
