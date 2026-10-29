# Methodology

The unit of observation is a station on a calendar day. RainTomorrow is mapped Yes→1, No→0. The only removed rows have an unknown target or belong to stations outside the chosen region. No complete-case filter is applied to predictors.

Date is retained for splitting, then replaced in the feature matrix by sine/cosine of day of year (365.25-day period). Median imputation and standardization apply to numeric features. Most-frequent imputation and one-hot encoding apply to categorical features; unknown categories are ignored. Every learned transformation is inside a scikit-learn Pipeline fitted separately within each CV training fold.

Training covers the earliest 70% of unique dates; validation the next 15%; test the remaining 15%. Percentages refer to dates, so row percentages differ where station coverage changes. The CV gap is one observed date, potentially more than one calendar day where observations are absent. Cross-validation scores use average precision to account for the minority rain class.

Logistic regression searches C ∈ {0.1, 1, 10} and class weights {none, balanced}. The forest searches maximum depth {8, unrestricted} and minimum leaf size {4, 12}, using 150 trees and balanced subsample weights. Random seeds are 42. A train-prior DummyClassifier provides a transparent baseline.

Validation selects between the best CV models and sets the F1 threshold. Ties follow the deterministic candidate order. The winning estimator stays fitted to training only: this keeps the validation-selected probability threshold attached to the estimator that produced those probabilities. The final test is not used to pick a model, hyperparameter, threshold or feature.

ROC AUC measures ranking; average precision emphasizes positive-class retrieval; Brier score measures probability error. Accuracy alone is insufficient because rain prevalence is about one quarter. No confidence interval is presented as if station-days were independent: nearby stations and adjacent days are correlated.

References: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
