
## Relevance rule and noise floor for spatially varying effects (2026-11-12, on hold)
- At global scale every extra component will be "significant"; decide by relevance, fixed before fitting:
  (1) size: spread of kappa(s) (10th-90th pct over cells with unrest) >= kappa itself, or sign changes over a substantial area;
  (2) withheld-data log score gain on cell-weeks with recent unrest larger than its SE;
  (3) interpretation: does the map change the story.
  Keep gamma(s) only if (1) and one of (2)/(3); same rule for regional kappa and lambda(s).
- Noise floor for (1): semi-synthetic null on the West Africa design (real cells/weeks/covariates, outcomes regenerated
  week by week from the fitted extended model with constant kappa), refit with gamma field, only gamma's 2 hyperparameters
  free; 10 replicates; ~0.5 day code + 1.5-3 h CRC. Per region later (floor depends on event density).
