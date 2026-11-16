"""Task 49 merge: boundary likelihood-ratio test of each country effect on escalation.

  python realdata/merge_tau.py <dir with ce_main.json and ce_tau_<o>_<t>.json> <out.json>

2 x (log ML full - log ML without the term), p = 0.5 P(chi2_1 > 2 diff) (tau = 0 is on the boundary; for a BYM2 term the
mixing phi is not identified under the null, so the p-value is approximate)."""
import glob
import json
import os
import sys

from scipy.stats import chi2

d, out = sys.argv[1], sys.argv[2]
full = json.load(open(os.path.join(d, "ce_main.json")))["models"]["biv3_cs"]
res = {"log_ml_full": full["log_ml_full"], "tests": {}}
for f in sorted(glob.glob(os.path.join(d, "ce_tau_*.json"))):
    r = json.load(open(f))["models"]["biv3_cs"]
    d2 = 2 * (full["log_ml_full"] - r["log_ml_without"])
    res["tests"][r["tau_drop"]] = {"log_ml_without": r["log_ml_without"], "two_diff": d2, "opt_success": r["opt_success"],
                                   "p_boundary": float(0.5 * chi2.sf(max(d2, 0.0), 1))}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
