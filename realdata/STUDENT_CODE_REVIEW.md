# Notes on the original pipeline (papersblindreview/26.Conflicts.unpublished)

Read from the public code on 2026-10-31. The data are not in that repository, so items marked
**check** could not be verified numerically yet. Each one can be checked with one line once
the partition files `data/partitions/data_final_world_XX.csv` are available.

## 1. Border cells are duplicated and get mixed national covariates (likely serious)

`prepare_sustainability`, `prepare_EPI`, `prepare_gdp` and `prepare_vdem` (scripts/functions.py)
join country polygons to grid cells with `sjoin(..., predicate='intersects')` and return one
row per `(ISO_A3, grid_id, year)`. A cell that touches *k* countries therefore has *k* rows.
`stage_dataset` (scripts/preprocessing.py) then drops `ISO_A3` and merges each table on
`['grid_id', 'year']`:

- every merge multiplies the rows of a border cell by *k*, so after sustainability, EPI, GDP
  and V-Dem a cell on a two-country border appears up to 2^4 = 16 times per week, and
  IPC regions (`adm1_pcode`) multiply it further;
- the duplicates combine indicators from *different* countries (country A's GHG with country
  B's EPI, and so on);
- each duplicate carries the same outcome and lags, so border cells are heavily over-weighted
  in training and in the SHAP explanation sample;
- `country_pop` used to apportion GDP is summed over the duplicated rows, so GDP per cell is
  wrong for every country with a border cell.

This would explain why the held-out set has 15.36M rows for about 105 weeks, i.e. about
146,000 rows per week, against roughly 65,000 land cells at 0.5°.

**Check:** `data.groupby(['grid_id','week_start']).size().describe()` should be 1 everywhere.
**Fix:** assign each cell to a single country (for example, the country covering the cell
centroid, or the largest overlap), keep `ISO_A3` in the final table, and merge national
indicators on `['ISO_A3', 'year']`.

## 2. Early stopping uses the evaluation years

`model_xgb.py` trains with `evals=[(dtrain,'train'), (dval,'val')]` and
`early_stopping_rounds=100`, where `dval` is 2023–2024, the same data used for the reported
AUC-PR. The number of trees is therefore tuned on the test set. The bootstrap then reuses
`best_iteration + 1`. **Fix:** stop early on 2022 (or on a block of training cells) and report
2023–2024 untouched.

## 3. Training rows are sampled with replacement

`np.random.choice(n, size=min(5e6, n))` samples with replacement by default, so about 37% of
the training rows are dropped and others are repeated. This is harmless for the fit, but it is
probably not what was intended (`replace=False`).

## 4. Countries outside ACLED coverage are coded as zero events

Every cell-week without an ACLED event is set to 0 (`fillna(0)`), including country-years
before ACLED covered the country. These are missing values, not zeros. **Fix:** drop cell-weeks
before each country's coverage start date.

## 5. Precipitation "anomaly" is global, not local

`ppt_anomaly` subtracts the training-period mean for the calendar month computed over *all
cells of the world*. The result is raw precipitation minus a global constant for each month,
not a departure from the local climate. Temperatures enter raw. The biome analysis therefore
mixes climatology and anomaly.

## 6. Smaller points

- The grid is every 0.5° cell that *intersects* Natural Earth land, so it includes coastal
  slivers and Antarctica. Cells whose biome is `N/A` (rock and ice) are dropped at model time.
- Cell centroids are computed in EPSG:3857, which is harmless at 0.5°.
- The 4-week rolling sums are computed with an ungrouped `.rolling()` after a grouped `shift`.
  Because the first lag of each cell is NaN, NaN propagation stops values from leaking across
  cells, but the first four weeks of every cell are missing.
- `scale_pos_weight = sqrt(n0/n1)` shifts the fitted log-odds by `log(sqrt(n0/n1))`. Predicted
  probabilities are not calibrated (see `sc/diagnostics.prior_correction`).

## What the spatial analysis needs

The partition files (or a Parquet export of them), ideally after fixing item 1 and keeping
`ISO_A3`. Split into files under 100 MB each, they fit in a private GitHub repository without
Git LFS. `realdata/load_partitions.py` reads either format, removes duplicated cell-weeks if
item 1 has not been fixed, and rebuilds outcomes and lags exactly as in the original code.
