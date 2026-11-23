# Lab 3: Ticket Escalation — Answers

## 1. Understanding the data
The target is **`escalated`**. A value of 1 means the ticket was escalated, and 0 means it was not. This is a **binary classification** problem.

The dataset has 900 rows. The target is imbalanced: about 20.7% of tickets are escalated and 79.3% are not. There are missing values in some columns, so I used imputation during preprocessing.

## 2. Models and test results
I used a 75% training and 25% testing split with `random_state=42` and stratification.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.796 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.822 | 0.750 | 0.196 | 0.310 | 0.811 |
| LDA | 0.822 | 0.750 | 0.196 | 0.310 | 0.809 |
| k-NN (k=15) | 0.796 | 0.500 | 0.087 | 0.148 | 0.705 |

The majority baseline has fairly high accuracy because most tickets are not escalated. But it misses every escalated ticket, so accuracy alone does not show how useful the model is. Logistic Regression and LDA performed best overall in these results.

## 3. Threshold comparison (Logistic Regression)

| Threshold | Precision | Recall | F1 | FP | FN | Cost (5×FN + FP) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.429 | 0.783 | 0.554 | 48 | 10 | 98 |
| 0.30 | 0.549 | 0.609 | 0.577 | 23 | 18 | 113 |
| 0.40 | 0.571 | 0.261 | 0.358 | 9 | 34 | 179 |
| 0.50 | 0.750 | 0.196 | 0.310 | 3 | 37 | 188 |
| 0.60 | 1.000 | 0.065 | 0.122 | 0 | 43 | 215 |
| 0.70 | 0.000 | 0.000 | 0.000 | 0 | 46 | 230 |

Lowering the threshold usually increases recall because more tickets are predicted as escalated. It also increases false positives and lowers precision. The lowest cost in this test table is at threshold 0.20. Choosing the threshold using the test set can make the result look better than it will be in real use, so the final threshold should be selected using validation data and then checked on a separate test set.

## 4. Cross-validation
I used stratified 5-fold cross-validation on the training data to compare k values 5, 9, 15, 21, and 31 using F1. The best result was **k=5**, with mean F1 around **0.243**.

## 5. Basic probability answers
- **Prior:** the probability of a class before looking at the ticket features.
- **Likelihood:** how likely the observed features are if a particular class is true.
- **Posterior:** the updated probability of a class after considering the features.
- **Evidence:** the overall probability of observing those features.

If the probability of escalation is 0.30, false-positive cost is 1, and false-negative cost is 5, then predicting escalation has expected cost (1 × 0.70 = 0.70). Predicting no escalation has expected cost (5 × 0.30 = 1.50). So escalation is the lower-cost decision.

## 6. Recommendation
I would start with Logistic Regression because it performed well and its probabilities can be used to adjust the decision threshold. Since missing an escalated ticket is much more costly, I would consider a lower threshold than 0.50, then choose it using validation data. I would monitor false negatives, false positives, precision, and recall after deployment. I would also check performance across ticket priorities and channels, and retrain if the data changes.
