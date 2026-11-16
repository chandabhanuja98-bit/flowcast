# Model selection and evaluation summary

All scores are on the time-ordered split: train 2025-01-08 to 2025-04-18, validation 2025-04-18 to 2025-05-10, test 2025-05-10 to 2025-05-31. Winners are chosen on **validation**; the test window is only reported.

## 1. PRD success criteria

| Criterion | Target | Result | Status |
|---|---|---|---|
| Volume forecast accuracy | MAPE <= 12% | 8.6% (XGBoost), 8.7% (GRU) | Met |
| Deep model vs best classical on volume RMSE | GRU/LSTM lower | GRU 53.69 vs XGBoost 53.93 (test); 51.50 vs 55.21 (validation) | Met, narrowly on test |
| Congestion classification | macro-F1 >= 0.80 | 0.771 (Random Forest, tuned GRU) | Not met |
| Accident-risk ranking | ROC-AUC >= 0.75 | 0.642 (GRU), 0.595 (best classical) | Not met |

## 2. Winning classical model per target (PRD 11.4)

| Target | Winner | Validation | Test | Runner-up (validation / test) |
|---|---|---|---|---|
| Volume | XGBoost | RMSE 55.21 | RMSE 53.93, MAPE 8.60% | Random Forest 58.48 / 55.00 |
| Travel time | XGBoost | RMSE 1.020 | RMSE 1.046, MAPE 9.11% | Random Forest 1.042 / 1.079 |
| Congestion | Random Forest | macro-F1 0.764 | macro-F1 0.771 | XGBoost 0.759 / 0.764 |
| Accident risk | Decision Tree | ROC-AUC 0.545 | ROC-AUC 0.595, recall 0.393 | XGBoost 0.535 / 0.570 |

Why these models beat the alternatives on the criteria that matter operationally:

- **Volume and travel time, XGBoost.** Lowest validation RMSE for both, ahead of Random Forest by 5% on volume. Validation and test scores agree (volume 55.2 and 53.9), so the ranking is not a quirk of one window. Prediction for the whole 25,925-row test window takes 0.016 s (volume) and 0.032 s (travel time), well inside the 30 s batch budget.
- **Congestion, Random Forest.** Best macro-F1 on validation (0.764) and test (0.771), and the best accuracy (0.874). Its confusion matrix shows the errors are neighbouring classes (mostly Heavy and Severe), which an operator can interpret. Inference takes 0.047 s. Probability outputs give the congestion confidence used by the dashboard. The tuned variant adds nothing (0.770 on test), so the plain model is used.
- **Accident risk, Decision Tree.** Highest validation ROC-AUC (0.545) and test ROC-AUC (0.595), and the highest recall at the default threshold (0.393, against 0.016 for Random Forest, which almost never raises an alert). It is a single tree, so its rules can be read off directly, and it predicts in 0.003 s. All classical models are close to chance on this target, so the choice is a small margin; see section 4.
- **Fold stability.** Cross-validation fold scores were used inside the grid search but were not stored, so stability is judged here by agreement between validation and test scores.

## 3. Classical vs deep model, same test window (PRD 12.4)

| Model | Volume RMSE | Volume MAPE | Congestion macro-F1 | Accident ROC-AUC |
|---|---|---|---|---|
| Linear Regression | 74.92 | 14.66% | - | - |
| Decision Tree | 58.95 | 9.56% | 0.753 | 0.595 |
| Random Forest | 55.00 | 8.74% | **0.771** | 0.586 |
| XGBoost | 53.93 | **8.60%** | 0.764 | 0.570 |
| SVM (baseline) | - | - | 0.758 | 0.518 |
| LSTM | 54.07 | 8.83% | 0.767 | 0.631 |
| **GRU** (seq 36, hidden 96, 2 layers, dropout 0.4) | **53.69** | 8.69% | 0.769 (0.771 tuned) | **0.642** |

Did the deep model earn its added complexity? Partly.

- On volume RMSE the GRU beats the best classical model, by 0.45% on test and 6.7% on validation. XGBoost still has the lower MAPE (8.60% against 8.69%). The gain is real but small, and the GRU needs 20 to 30 minutes to train against about a minute for XGBoost.
- On congestion the GRU matches Random Forest (0.771) and does not exceed it.
- On accident risk the GRU is the best model (0.642 against 0.595), a gain of 0.047, but still far from the 0.75 target.
- The LSTM did not beat XGBoost on volume (54.07 against 53.93).

## 4. Why congestion macro-F1 0.80 and accident ROC-AUC 0.75 were not reached

These findings come from exploratory checks run outside the pipeline code (analysis scripts, not part of `src/`).

**Congestion.** The class is defined from the next window's volume divided by capacity (raw labels follow this rule 99.95% of the time), and the Heavy band is only 0.10 of capacity wide.

- The volume error left after modelling is pure noise: its correlation with the previous window's error is 0.013, and across roads at the same time 0.015.
- The same road, time slot and weekday varies about 10% from week to week on calm days. A typical-traffic profile alone is off by about 11%, the same as the full models (11.1% on ordinary days).
- A simulated perfect-mean model with that noise reaches about 0.81 (10% noise) to 0.77 (12% noise). Realistic best is about 0.775 to 0.78.

Attempts that all stopped near the same ceiling:

| Attempt | Best congestion macro-F1 |
|---|---|
| Four classical model types (SVM, Tree, Forest, XGBoost) | 0.771 |
| Trend features | 0.763 |
| Class-weight tuning on validation | 0.770 |
| Weekly lag, next-slot profile, network features | 0.771 |
| Predict the volume ratio, then convert to classes | 0.772 |
| Larger models, averaging, refit on train+validation | 0.774 |
| Prediction as deviation from a pooled profile | 0.769 |
| GRU / LSTM | 0.771 / 0.768 |

**Accident.** Only 0.9% of windows have an accident in the next window. Rain (2.8% against 0.8%), low visibility (2.9%) and peak hours (1.3% against 0.7%) raise the rate, but no single feature exceeds about 0.58 AUC. Accident counts are spread evenly over the 25 roads (45 to 77 each). A model given the same window's features exactly (information a forecaster does not have) reaches ROC-AUC 0.642, which is where the GRU already is, so 0.75 is out of reach on this data. Accident-history features did not help (0.564 to 0.577).

## 5. Persisted models

Every saved model has a model card (training window, features, hyperparameters, test metrics) beside it in `models/`. Classical models are reloaded with `utils.load_model`; the GRU and LSTM with `dl_model.load_forecaster`. Predictions for the validation and test windows, with 90% intervals for volume and travel time and a probability-based confidence for congestion, are in `data/processed/predictions_v1.csv` (classical winners) and `predictions_gru_v1.csv` (GRU).
