# Standing rules for this project

## Evidence for every model component
Every latent process or fixed effect in a model we report must come with evidence that it is
needed, and the Supplementary Material must present that evidence. This applies to the
components already in the model as well as to new ones.

- **Variance components and latent fields** (country effects u, cell effects e, static field w,
  space-time fields v, and any new field). For a variance component, "interval excludes zero"
  is the wrong test: zero is the boundary, so the test cannot work. Report both of these:
  1. the log marginal likelihood of the model without that component, refitted on the same
     subsample;
  2. an approximate interval for log sigma (and for the range and AR coefficient), from the
     Hessian of the Laplace log marginal likelihood at the mode.
- **Fixed effects and interaction-type parameters** (delta, kappa, neighbour terms, the
  loading lambda, escalation psi): report the estimate with its SE or interval, and say
  whether the interval excludes zero.
- A component without evidence is dropped, or kept and stated as a control with no
  interpretation.
- A good story is not evidence. Do not justify a component by its interpretation alone.
