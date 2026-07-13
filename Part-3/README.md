# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

Project Objective

In this part, I worked with different tree-based machine learning models and compared their performance.

The main goal was to:

- Train a Decision Tree model.
- Reduce overfitting by controlling the tree.
- Compare Gini and Entropy.
- Train Random Forest.
- Train Gradient Boosting.
- Perform Feature Ablation.
- Compare models using Cross Validation.
- Tune the Random Forest using GridSearchCV.
- Save the best model for future use.

---

# Dataset

Dataset: House Prices Dataset

Final Feature Shape:

```
(1460, 215)
```

The dataset was already cleaned and preprocessed in Part 2.

---

# Step 1 - Decision Tree (Default)

First I trained a normal Decision Tree without giving any limits.

Since there was no restriction on the tree depth, the model kept learning every small pattern from the training data.

### Results

| Metric | Value |
|---------|-------|
| Training Accuracy | 1.0000 |
| Testing Accuracy | 0.8904 |

### Observation

The model got 100% training accuracy but much lower testing accuracy.

This is a clear sign of **overfitting**.

Decision Trees are called **high variance models** because they make greedy decisions at every split. Once a split is made, the tree never goes back to change it. Because of this, even small changes in the training data can create a completely different tree.

---

# Step 2 - Controlled Decision Tree

To reduce overfitting, I limited the tree by using:

- max_depth = 5
- min_samples_split = 20

These parameters stop the tree from becoming too deep and prevent it from creating very small splits.

### Results

| Metric | Value |
|---------|-------|
| Training Accuracy | 0.9307 |
| Testing Accuracy | 0.8801 |
| Accuracy Gap | 0.0505 |

The training accuracy became lower, but the model became simpler and less likely to memorize the training data.

---

# Gini vs Entropy

I also compared two splitting criteria.

### Formula

### Gini Impurity

```
Gini = 1 − Σ(pi²)
```

### Entropy

```
Entropy = −Σ(pi log₂ pi)
```

If **Gini = 0**, it means every sample inside that node belongs to only one class, so the node is perfectly pure.

### Results

| Criterion | Test Accuracy |
|------------|---------------|
| Gini | 0.8938 |
| Entropy | 0.8836 |

Both methods performed similarly, but Gini was slightly better on this dataset.

---

# Step 3 - Random Forest

Next I trained a Random Forest model.

Instead of using only one tree, Random Forest creates many decision trees and combines their predictions.

### Parameters

- n_estimators = 100
- max_depth = 10
- random_state = 42

### Results

| Metric | Value |
|---------|-------|
| Training Accuracy | 0.9949 |
| Testing Accuracy | 0.9418 |
| ROC-AUC | 0.9834 |

---

## Top 5 Important Features

| Feature | Importance |
|----------|-----------:|
| GrLivArea | 0.083388 |
| OverallQual | 0.056604 |
| FullBath | 0.053926 |
| YearBuilt | 0.049113 |
| BsmtQual | 0.049090 |

---

## How Feature Importance Works

Random Forest calculates feature importance by checking how much each feature reduces the **Gini Impurity** across all trees.

A feature that creates better splits gets a higher importance score.

This is different from Linear Regression because regression coefficients only show the relationship between input and output, while feature importance measures how useful a feature is during tree splitting.

---

## Bagging

Random Forest uses **Bagging (Bootstrap Aggregation)**.

Each tree is trained on a random sample of the training data with replacement.

Also, at every split, the model only checks a random subset of features instead of using all features.

Finally, the predictions of all trees are combined.

Because many trees work together, Random Forest reduces variance and usually performs better than a single Decision Tree.

---

# Step 4 - Gradient Boosting

Gradient Boosting works differently from Random Forest.

Instead of building all trees independently, it builds one tree at a time.

Every new tree tries to correct the mistakes made by the previous trees.

### Parameters

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

### Results

| Metric | Value |
|---------|-------|
| Training Accuracy | 0.9880 |
| Testing Accuracy | 0.9521 |
| ROC-AUC | 0.9863 |

---

# Feature Ablation Study

I removed the five least important features from the Random Forest model.

Removed Features

- Electrical_Mix
- SaleType_ConLI
- MiscFeature_TenC
- SaleCondition_AdjLand
- SaleType_Oth

### Results

| Model | ROC-AUC |
|--------|---------|
| Full Model | 0.9834 |
| Reduced Model | 0.9861 |

### Observation

The reduced model performed almost the same and even got a slightly better ROC-AUC score.

This shows that these five features were contributing very little.

Removing unnecessary features can make the model simpler without reducing performance.

In production, a simpler model is easier to maintain and can also reduce prediction time.

---

# Step 5 - Cross Validation

Instead of trusting only one train-test split, I used **5-Fold Stratified Cross Validation**.

This gives a more reliable estimate because every sample gets a chance to be used for testing.

### Results

| Model | Mean AUC | Std |
|--------|---------:|----:|
| Logistic Regression | 0.9525 | 0.0113 |
| Decision Tree | 0.9233 | 0.0148 |
| Random Forest | 0.9784 | 0.0051 |
| Gradient Boosting | 0.9749 | 0.0078 |

Random Forest achieved the highest average ROC-AUC.

---

# Step 6 - Hyperparameter Tuning

I used GridSearchCV to find the best Random Forest parameters.

The pipeline included:

- SimpleImputer
- StandardScaler
- RandomForestClassifier

Parameter Grid

- n_estimators = 50, 100, 200
- max_depth = 5, 10, None
- min_samples_leaf = 1, 5

Total combinations:

```
3 × 3 × 2 = 18
```

Using 5-fold Cross Validation:

```
18 × 5 = 90 Model Fits
```

### Best Parameters

```
max_depth = None
min_samples_leaf = 1
n_estimators = 200
```

### Best Cross Validation ROC-AUC

```
0.9796
```

---

# Manual Learning Curve

| Training Fraction | Training AUC | Testing AUC |
|------------------:|-------------:|------------:|
| 20% | 1.0000 | 0.9804 |
| 40% | 1.0000 | 0.9823 |
| 60% | 1.0000 | 0.9841 |
| 80% | 1.0000 | 0.9827 |
| 100% | 1.0000 | 0.9851 |

### Observation

Training AUC stayed at 1.0 because Random Forest can fit the training data very well.

Testing AUC improved slightly as more training data was used.

This suggests that adding more data may still improve performance a little.

---

# Step 7 - Model Serialization

Finally, I saved the best model using Joblib.

```
joblib.dump(best_pipeline, "best_model.pkl")
```

Then I loaded the saved model again.

```
joblib.load("best_model.pkl")
```

The model loaded successfully.

Predictions

```
[0 1]
```

Prediction Probabilities

```
[0.03 1.00]
```

This confirms that the saved model works correctly.

---

# Final Model Comparison

| Model | Test AUC | 5-Fold Mean AUC |
|--------|---------:|----------------:|
| Logistic Regression | - | 0.9525 |
| Decision Tree | - | 0.9233 |
| Random Forest | 0.9834 | 0.9784 |
| Gradient Boosting | 0.9863 | 0.9749 |
| Tuned Random Forest | 0.9851 | 0.9796 |

---

# Final Recommendation

If I had to choose one model for this project, I would recommend the **Tuned Random Forest**.

It gave the best cross-validation score and also performed very well on the test data.

Random Forest is also stable because it combines many trees instead of depending on only one.

Gradient Boosting also performed really well, but Random Forest was easier to tune and gave more consistent results.

---

# Files Included

- Part3.ipynb
- best_model.pkl
- README.md

---

# Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

# Conclusion

In this part, I learned how different tree-based machine learning models work.

I compared Decision Tree, Random Forest and Gradient Boosting.

I also learned how to reduce overfitting, compare models using Cross Validation, tune hyperparameters using GridSearchCV, perform Feature Ablation, and save the final trained model.

Overall, Random Forest and Gradient Boosting performed much better than a single Decision Tree on this dataset.

---

This completes Part 3 of the project.
