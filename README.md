# Bank Customer Churn Prediction

A classification model that predicts which Beta Bank customers are likely to leave, built to meet a strict F1 ≥ 0.59 target on held-out test data — with two class-imbalance correction techniques compared head-to-head.

## Business Question

Beta Bank was losing customers gradually and wanted to act before they left, since retaining an existing customer is cheaper than acquiring a new one. The task was to build a model that flags at-risk customers with a minimum F1-score of 0.59 on the test set, and to compare it against AUC-ROC.

## Data

10,000 customers with credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, activity status, and estimated salary. Target: whether the customer exited (`Exited`). Classes are imbalanced — 80.1% stayed vs. 19.9% left.

## Approach

1. **Preprocessing** — dropped non-predictive identifiers (row number, customer ID, surname), imputed the ~9% of missing `Tenure` values with the median, one-hot encoded `Geography` and `Gender` (with `drop_first` to avoid the dummy variable trap), split data 3:1:1 (train/validation/test), and scaled numeric features with `StandardScaler` fit only on the training set
2. **Baseline model** — trained a Random Forest with no imbalance handling, to measure the cost of ignoring the class skew
3. **Imbalance correction** — compared two techniques on the validation set, sweeping `max_depth` for each:
   - Class weighting (`class_weight='balanced'`)
   - Upsampling the minority class (tested repeat factors of 2×, 3×, 4×)
4. **Final test evaluation** — retrained the best configuration and scored it once on the untouched test set, reporting both F1 and AUC-ROC

## Key Findings

- **The unweighted baseline missed the target**: F1 = 0.577, below the 0.59 minimum — the model was biased toward the majority class (customers who stayed), as expected with imbalanced data
- **Both correction techniques helped substantially.** Upsampling (3× repeat, `max_depth=8`) edged out class weighting on validation — F1 = 0.635 vs. 0.622. Deeper trees (unlimited depth) made results worse in both cases, a sign of overfitting to the expanded majority-adjacent data
- **Final test result: F1 = 0.603** (above the 0.59 target) **and AUC-ROC = 0.859**
- **The F1/AUC-ROC gap is informative**: the much higher AUC-ROC shows the model separates the two classes well across all thresholds, while F1 — evaluated at the default 0.5 cutoff — is more sensitive to that specific decision boundary. This suggests threshold tuning could likely push F1 even higher without changing the model itself

## Tools

Python · pandas · NumPy · scikit-learn (`RandomForestClassifier`, `StandardScaler`, `f1_score`, `roc_auc_score`) — class weighting and upsampling for imbalance correction

## Files

- `customer_churn_prediction.ipynb` — full analysis: preprocessing, baseline, imbalance correction comparison, and final test evaluation

---
*Coursework project completed as part of the Data Science MBA program.*
