# Spatial latent-field extensions of the ACLED conflict/unrest model

Code and simulation studies for turning the XGBoost + SHAP analysis of
Menicali, Marcantonio & Castruccio ("Detecting Antecedents of Violence and
Stanchions of Peace Across Seasons and Biomes") into a statistical model in
which the residual risk is modelled with spatial / space-time latent fields.

Everything is in Python, with no R-INLA dependency. `sc/laplace.py` is a small,
INLA/TMB-style empirical-Bayes Laplace engine for latent Gaussian models with
Bernoulli, binomial or Poisson likelihoods.

## Layout

| path | content |
|---|---|
| `sc/gmrf.py` | latent blocks: fixed effects, iid (country) effects, **Matérn SPDE field on the sphere** (finite-volume discretisation on a lon/lat grid, distances in km, masks, longitude wrap-around), separable **AR(1) × Matérn space-time** field; PC priors |
| `sc/laplace.py` | `LatentGaussianModel`: Newton mode, Laplace marginal likelihood, hyperparameter optimisation, Gaussian posterior sampling, predictions |
| `sc/simulate.py` | simulator mimicking the cell × week data: countries, biomes, climate anomalies, lags, national indicators, latent fields, optional spatial confounding |
| `sc/boosting.py` | XGBoost with the paper's hyperparameters, **spatially cross-fitted offsets**, exact TreeSHAP, SHAP slopes, GPBoost wrapper |
| `sc/diagnostics.py` | scores (log score, Brier, AUC-PR, AUC-ROC), persistence baseline, prior correction for `scale_pos_weight`, randomised quantile and cell Pearson residuals, Moran's I, great-circle variogram, residual ACF, within- vs across-border residual correlation |
| `sims/s1_residual_field.py` … `s4_svc.py` | simulation studies (below) |
| `realdata/load_partitions.py` | builds one clean row per cell-week from the original partition files (repairs duplicated border cells, assigns countries) |
| `realdata/reproduce_xgb.py` | reruns the original XGBoost model as published and in corrected form |
| `realdata/fit_acled.py` | hybrid spatial model on the real data: cross-fitted XGBoost offset + country effects + barrier Matérn field |
| `realdata/STUDENT_CODE_REVIEW.md` | issues found in the original pipeline |
| `crc/` | task list, SGE job scripts and status log for the runs on the Notre Dame CRC |
| `results_real/` | summaries of the real-data runs (no ACLED rows) |
| `results/` | outputs of the simulation studies |

## Running

```bash
pip install numpy scipy pandas scikit-learn xgboost matplotlib joblib pyarrow gpboost pymetis
```

```bash
cd sims && python s1_residual_field.py      # N_REPS=20 by default
```

```bash
cd sims && python s2_confounding.py         # N_REPS=30, B_BOOT=20
```

```bash
cd sims && python s3_persistence.py         # N_REPS=20, FIT_ST=1
```

```bash
cd sims && python s4_svc.py                 # N_REPS=30, B_BOOT=10
```

```bash
python realdata/fit_acled.py data.parquet --outcome conflict --region africa
```

```bash
python paper/make_tables.py                 # LaTeX tables for the paper from results/
```

Studies checkpoint every replicate to `results/*_ckpt.jsonl` and resume if interrupted.

## Simulation results (summary)

| Study | Question | Main finding |
|---|---|---|
| S0 | Does the engine work? | Range 399 km (truth 400), sd 0.99 (truth 1), 95% coverage 0.92 (fixed effect) / 0.94 (field). Edge cells have 2x variance (reflecting boundary); this motivates the barrier model for coastlines. |
| S1 | Residual structure after XGBoost | Moran's I of residuals 0.33 (paper-style XGBoost) / 0.59 (cross-fitted). Adding a Matérn field beats paper-style XGBoost in held-out log score in 20/20 replicates (AUC-PR 0.235 → 0.245; oracle 0.267). |
| S2 | National indicators (EPI-like) | Without confounding, naive SEs are 6x too small; the paper's block-bootstrap IQR criterion flags a null effect in 70% of replicates. With confounding, every estimator is biased; XGBoost and the GLM get the **sign wrong** for a true protective effect; the spatial field and spatial+ recover the sign but only 1/3–1/2 of the size. |
| S3 | Does recent history mean violence begets violence? | Under pure heterogeneity (true delta = 0): lags-only delta-hat = 2.13, + static field 0.70, + space-time field 0.11. XGBoost lag importance is *higher* when delta = 0. |
| S4 | Biome-specific climate effects | SVC model: coverage 0.90–0.97, cor(beta-hat(s), beta(s)) = 0.92. XGBoost SHAP slopes attenuated about 3x; the paper's median-SHAP-by-biome is about −0.045 in every biome regardless of the truth, and its IQR excludes zero in 93–100% of replicates. |

## Paper

`paper/` contains the AoAS draft (`main.tex`, `imsart` class). It has Data, Methods and
Simulation sections; Introduction, Application and Discussion are placeholders.
Upload to Overleaf together with `imsart.cls` and `imsart-nameyear.bst`.

