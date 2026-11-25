# Experiments log

Every experiment, including the ones that didn't help. Cross-validation (CV) is repeated stratified 5-fold × 5 repeats on the 891 training passengers, unless noted. Kaggle scores are accuracy on all 418 test passengers.

## 1. Exploration
- Sex was the strongest signal: 74% of women survived vs 19% of men.
- Class mattered: 63% in 1st class, 47% in 2nd, 24% in 3rd.
- Family size: people travelling alone (30%) and large families of 5+ fared worse than families of 2–4 (55–72%).
- Missing data: Age 19.9%, Cabin 77.1%, Embarked 0.2%.

## 2. v1: Random Forest baseline
- **Features:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize.
- **Pipeline:** median imputation for numbers, most-frequent imputation + one-hot encoding for categories, all inside a scikit-learn `Pipeline`.
- Compared logistic regression (79.2%), decision tree (80.6%) and random forest (82.7%) with a single 5-fold CV; random forest won.
- **Data leak found:** a grid search fitted on all 891 rows, then scored on the 179-row validation split, gave 89.94%. The validation rows had already been seen. Retraining on the 712 training rows only gave an honest **80.45%**.
- Final v1 settings: 200 trees, depth 7, balanced class weights.
- **Kaggle: 0.77511** (324/418). Repeated CV: 81.7%.

## 3. v2: Feature engineering
- **Added:** Title from the name (Mr, Mrs, Miss, Master, Rare), missing ages filled with the median for each title, HasCabin.
- Settings chosen by repeated CV, not the single validation split:

| Variant | CV |
|---|---|
| v1 features, depth 7, balanced | 81.7% |
| v1 features, depth 5 | 82.2% |
| v2 features, depth 5 | **83.0%** |
| v2 features, depth 5, balanced | 82.6% |
| v2 features, depth 7 | 82.7% |
| v2 features, logistic regression | 82.7% |
| v2 features, gradient boosting | 83.3% |

- **Kaggle: 0.77272** (323/418), one passenger fewer than v1.
- **Lesson:** a +1.3 point CV gain, inside a ±2% spread between folds, did not carry over to the test set.

## 4. Family-group rule (final)
- Start from the gender rule (Kaggle 0.76555, 320/418).
- Group passengers into families by surname, class, ticket without its last digit, port and fare. Passengers not matched by surname are linked by ticket.
- Using only women and boys in the training data:
  - all of a family's women and boys died → that family's women in the test set are predicted to die (9 changes);
  - all survived → that family's boys are predicted to survive (8 changes).
- CV: **84.1%**, vs 78.7% for the gender rule alone. The rules are rebuilt from the training folds only.
- **Kaggle: 0.80143** (335/418). 17 predictions changed; 16 right and 1 wrong.
- Idea credited to Chris Deotte's Titanic notebooks on Kaggle.

## Ideas not yet tried
- Fare per person (fare divided by the number of passengers on the ticket) and ticket-group size.
- Sex × Pclass interaction features.
- A soft-voting ensemble of gradient boosting, random forest and logistic regression, averaged over several seeds.
- Combining the family-group rule with a model for passengers it doesn't cover.
