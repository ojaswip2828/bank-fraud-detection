# Model Report

Time-based split. Threshold tuned on validation for recall >= 0.9.

| model | ROC-AUC | PR-AUC | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.9643 | 0.2987 | 0.0334 | 0.9143 | 0.0645 | 5554 | 18 |
| random_forest | 0.981 | 0.6657 | 0.0854 | 0.9 | 0.1559 | 2025 | 21 |

**Best by PR-AUC:** `random_forest`

### Recall by fraud type (best model)

| fraud_type     |   caught |   total |   recall |
|:---------------|---------:|--------:|---------:|
| card_testing   |      120 |     120 |    1     |
| takeover       |       50 |      65 |    0.769 |
| unusual_amount |       19 |      25 |    0.76  |

## Ablation: removing the strongest shortcut features

Same split, same threshold rule; `pr_auc_drop` is vs the full model.

| features                         | model               |   roc_auc |   pr_auc |   precision |   recall |   false_alerts |   pr_auc_drop |
|:---------------------------------|:--------------------|----------:|---------:|------------:|---------:|---------------:|--------------:|
| full model                       | logistic_regression |    0.9643 |   0.2987 |      0.0334 |   0.9143 |           5554 |        0      |
| full model                       | random_forest       |    0.981  |   0.6657 |      0.0854 |   0.9    |           2025 |        0      |
| no category                      | logistic_regression |    0.9565 |   0.3586 |      0.0329 |   0.8857 |           5468 |        0.0599 |
| no category                      | random_forest       |    0.9619 |   0.6577 |      0.0285 |   0.8952 |           6412 |       -0.008  |
| no is_new_merchant               | logistic_regression |    0.9435 |   0.1966 |      0.0307 |   0.8714 |           5773 |       -0.1021 |
| no is_new_merchant               | random_forest       |    0.9767 |   0.6332 |      0.0556 |   0.9143 |           3259 |       -0.0325 |
| no category + no is_new_merchant | logistic_regression |    0.9328 |   0.2211 |      0.0219 |   0.8524 |           8010 |       -0.0776 |
| no category + no is_new_merchant | random_forest       |    0.9398 |   0.608  |      0.0136 |   0.9    |          13745 |       -0.0577 |

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
Top features (mean |SHAP|, fraud-probability units, 500 test rows):

| feature | mean abs SHAP |
|---|---|
| log_amount | 0.0889 |
| mins_since_prev | 0.0844 |
| is_new_merchant | 0.0793 |
| amt_to_hist_avg | 0.0448 |
| category | 0.0387 |
| txns_last_1h | 0.0377 |
| channel | 0.0357 |
| is_out_of_city | 0.0302 |
| amt_last_24h | 0.0301 |
| is_night | 0.0220 |
