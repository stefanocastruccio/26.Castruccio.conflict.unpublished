# Standing rules for this project

## 95% credible intervals for every estimated parameter
Whenever we estimate parameters (fixed effects and other parameters of interest), report them with
95% credible intervals, as long as computing the intervals does not substantially increase the cost.
- Parameters that are part of the narrative go in the main text, with their intervals.
- All others go in the Supplementary Material, with their intervals.
Each model we report must justify its components through this uncertainty.

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
