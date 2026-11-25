# Leaderboard history

Every file submitted to Kaggle, in order. Kaggle's Titanic score is accuracy on all 418 test passengers, so each passenger is worth about 0.24 points (1/418).

| # | Date (UTC+8) | File in this repo | Approach | Public score | Correct / 418 | Change |
|---|---|---|---|---|---|---|
| 1 | 2026-11-26 | `submissions/v1_random_forest.csv` | Random Forest, depth 7, balanced class weights | 0.77511 | 324 | first submission |
| 2 | 2026-11-26 | `submissions/v2_random_forest.csv` | Random Forest + Title, age by title, HasCabin, depth 5 | 0.77272 | 323 | −1 passenger |
| 3 | 2026-11-26 | `submissions/family_groups.csv` | Family-group (woman-child-group) rule | **0.80143** | **335** | +12 passengers, current best |
| 4 | 2026-11-26 | `submissions/blend_plus_family_rule.csv` | Blend of RF + XGBoost + LR probabilities, with family-rule overrides | 0.78229 | 327 | −8 vs best |

Submission 4 had the best cross-validation score (84.47% vs 84.13%) but scored 8 passengers fewer than the family rule. It differed from the family-rule file on 18 test passengers, and the family rule was right on 13 of them. The tiny CV edge did not hold.

Reference: Kaggle's example file (`gender_submission.csv`, all women survive) scores 0.76555 (320/418).

## Not yet submitted
Files in `submissions/candidates_not_submitted/` have cross-validation results only (see `results/model_comparison.csv`). A Kaggle score will be added here once a file is actually submitted.
