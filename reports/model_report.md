# Model Report

Time-based split. Threshold tuned on validation for recall >= 0.9.

| model | ROC-AUC | PR-AUC | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.9965 | 0.7731 | 0.3099 | 0.9159 | 0.4631 | 461 | 19 |
| random_forest | 0.9996 | 0.9279 | 0.7003 | 0.9513 | 0.8068 | 92 | 11 |

**Best by PR-AUC:** `random_forest`

### Recall by fraud type (best model)

| fraud_type     |   caught |   total |   recall |
|:---------------|---------:|--------:|---------:|
| card_testing   |      117 |     118 |    0.992 |
| takeover       |       69 |      77 |    0.896 |
| unusual_amount |       29 |      31 |    0.935 |

### logistic_regression
Top features (coefficient):

| feature | value |
|---|---|
| cat__category_electronics | 3.9256 |
| cat__category_travel | 3.8854 |
| cat__category_online_retail | 3.8689 |
| cat__category_atm_withdrawal | -3.6443 |
| cat__category_utilities | -3.4594 |
| cat__category_restaurant | -3.4211 |
| cat__category_fuel | -3.2132 |
| cat__channel_ATM | -3.1360 |
| cat__category_grocery | -3.0944 |
| cat__channel_POS | -2.8645 |

### random_forest
Top features (importance):

| feature | value |
|---|---|
| num__is_new_merchant | 0.2082 |
| cat__channel_ONLINE | 0.1277 |
| num__mins_since_prev | 0.1073 |
| num__log_amount | 0.1067 |
| num__txns_last_1h | 0.0958 |
| num__amt_to_hist_avg | 0.0674 |
| num__amt_last_24h | 0.0654 |
| num__is_out_of_city | 0.0617 |
| cat__category_online_retail | 0.0486 |
| cat__channel_POS | 0.0301 |
