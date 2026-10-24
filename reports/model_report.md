# Model Report

Time-based split. All models isotonic-calibrated. Threshold chosen for recall >= 0.9 on out-of-time predictions pooled over 3 expanding folds inside train, then applied once to the untouched test set.

`P@90R` = best precision reachable at 90% recall on test (threshold-free model comparison). `precision` / `recall` / `FP` / `FN` = what the chosen threshold actually gave.

| model | ROC-AUC | PR-AUC | P@90R | Brier | precision | recall | FP | FN |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.9643 | 0.262 | 0.0249 | 0.00313 | 0.0185 | 0.9524 | 10605 | 10 |
| random_forest | 0.9806 | 0.6579 | 0.0802 | 0.00181 | 0.0541 | 0.9286 | 3407 | 15 |
| lightgbm | 0.9867 | 0.7212 | 0.1294 | 0.00156 | 0.0642 | 0.9429 | 2886 | 12 |
| lightgbm (uncalibrated) | 0.9878 | 0.7415 | 0.1331 | 0.00271 | 0.0492 | 0.9476 | 3842 | 11 |

**Best by PR-AUC:** `lightgbm`

### Recall by fraud type (best model)

| fraud_type     |   caught |   total |   recall |
|:---------------|---------:|--------:|---------:|
| card_testing   |      120 |     120 |    1     |
| takeover       |       57 |      65 |    0.877 |
| unusual_amount |       21 |      25 |    0.84  |

## Ablation: removing the strongest shortcut features

Same split, same threshold rule; `pr_auc_drop` is vs the full model.

| features                         | model               |   roc_auc |   pr_auc |   precision_at_90_recall |   recall |   false_alerts |   pr_auc_drop |
|:---------------------------------|:--------------------|----------:|---------:|-------------------------:|---------:|---------------:|--------------:|
| full model                       | logistic_regression |    0.9643 |   0.262  |                   0.0249 |   0.9524 |          10605 |        0      |
| full model                       | lightgbm            |    0.9867 |   0.7212 |                   0.1294 |   0.9429 |           2886 |        0      |
| no category                      | logistic_regression |    0.9583 |   0.3437 |                   0.0182 |   0.8952 |           6081 |        0.0817 |
| no category                      | lightgbm            |    0.9628 |   0.6754 |                   0.0176 |   0.8952 |           7646 |       -0.0458 |
| no is_new_merchant               | logistic_regression |    0.9403 |   0.1838 |                   0.0164 |   0.919  |          15756 |       -0.0782 |
| no is_new_merchant               | lightgbm            |    0.9833 |   0.6609 |                   0.0814 |   0.9619 |           5695 |       -0.0603 |
| no category + no is_new_merchant | logistic_regression |    0.9244 |   0.1942 |                   0.007  |   0.9762 |          28877 |       -0.0678 |
| no category + no is_new_merchant | lightgbm            |    0.9487 |   0.609  |                   0.0214 |   0.8952 |           8031 |       -0.1122 |

## Global feature importance (SHAP)

### logistic_regression
Top features (mean |SHAP|, log-odds units, 500 test rows):

| feature | mean abs SHAP |
|---|---|
| category | 1.4679 |
| channel | 1.2491 |
| hist_txn_count | 0.8130 |
| is_new_merchant | 0.4289 |
| txns_last_1h | 0.3561 |
| is_out_of_city | 0.3038 |
| is_night | 0.2016 |
| amt_last_24h | 0.1919 |
| log_amount | 0.1883 |
| amt_to_hist_avg | 0.1489 |

### random_forest
Top features (mean |SHAP|, raw-score units, 500 test rows):

| feature | mean abs SHAP |
|---|---|
| log_amount | 0.0869 |
| mins_since_prev | 0.0865 |
| is_new_merchant | 0.0782 |
| amt_to_hist_avg | 0.0455 |
| category | 0.0396 |
| txns_last_1h | 0.0364 |
| channel | 0.0363 |
| is_out_of_city | 0.0304 |
| amt_last_24h | 0.0291 |
| is_night | 0.0220 |

### lightgbm
Top features (mean |SHAP|, log-odds units, 500 test rows):

| feature | mean abs SHAP |
|---|---|
| category | 1.3162 |
| log_amount | 0.9048 |
| amt_to_hist_avg | 0.6999 |
| channel | 0.4908 |
| mins_since_prev | 0.4403 |
| income_band | 0.4209 |
| is_new_merchant | 0.3487 |
| account_age_days | 0.3483 |
| hist_txn_count | 0.3084 |
| amt_last_24h | 0.2696 |
