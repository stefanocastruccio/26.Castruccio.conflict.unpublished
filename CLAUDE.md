# Standing rules for this project

## Credible intervals for every model component (always in the Supplementary Material)
Each model we report must justify its components through their uncertainty. The supplement must
always contain a table of credible intervals for the main effects and for every hyperparameter of
each reported model, in each region.

- **Fixed effects:** the estimate and a 95% interval, saying whether it excludes zero. Fixed
  effects are the coefficients of the histories (delta, kappa, the neighbour terms) and the main
  covariates.
- **Hyperparameters:** the estimate and a 95% interval. These are the SDs of the country effects,
  cell effects and static fields; the ranges; the AR coefficients; the loading lambda; and any new
  parameter such as the escalation coefficients psi. The intervals come from the Hessian of the
  Laplace log posterior of theta at the estimate (realdata/fit_bivariate.py --hessian-only,
  crc/jobs/33_hyper_ci.sh). For an SD, report how far the interval lies from zero.
- Whenever a model or region is added, compute its intervals before any of its results enter the
  paper.
- A component whose interval does not support it is dropped, or kept and stated as a control that
  we do not interpret.
- A good story is not evidence. Do not justify a component by its interpretation alone.
- Do not run ablation studies (refits without a component) unless the user asks for them.
