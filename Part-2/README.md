# Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate 

## Dataset

For this part, I used the `cleaned_data.csv` file that I created in Part 1.

The data was already cleaned, so I could directly start building machine learning models.

---

# Objectives

In this part I wanted to:

* Load the cleaned dataset
* Create regression and classification targets
* Encode categorical columns
* Split the data into training and testing data
* Scale the features
* Train regression models
* Train classification models
* Compare the models
* Check how well they perform

---

# Feature and Target

I used all columns except **SalePrice** and **Id** as my features.

For regression,

* Target = **SalePrice**

For classification,

I converted SalePrice into two classes using the median value.

If the house price is greater than the median price, I gave it class **1**.

Otherwise, I gave it class **0**.

This made the dataset almost balanced.

---

# Encoding Categorical Columns

Some columns already have an order.

For example,

* Poor
* Fair
* Typical
* Good
* Excellent

These values have a proper order, so I converted them into numbers.

I also did the same for columns like:

* BsmtExposure
* GarageFinish
* Functional
* Fence
* LotShape
* Utilities
* LandSlope
* PavedDrive

The remaining categorical columns do not have any order.

For those columns, I used One-Hot Encoding with `pd.get_dummies()` and `drop_first=True`.

I used one-hot encoding because it creates separate columns for each category and does not create a fake order between them.

After encoding, my dataset had **236 features**.

---

# Train Test Split and Scaling

I split the data into

* 80% Training
* 20% Testing

using `random_state = 42`.

After splitting, I used StandardScaler.

I fitted the scaler only on the training data and then transformed both training and testing data.

I did this because fitting the scaler on the whole dataset would cause data leakage.

If the scaler learns information from the test data before training, then the model evaluation will not be fair.

---

# Linear Regression

First, I trained a Linear Regression model.

### Results

* MSE = **2574508794.23**
* R² Score = **0.6644**

After training, I checked the model coefficients.

The largest coefficients were mostly related to Roof Material, Living Area, Second Floor Area and Overall Quality.

A large positive coefficient means that if a scaled feature increases by one unit, the predicted house price also increases by that coefficient value.

A large negative coefficient means that if a scaled feature increases by one unit, the predicted house price decreases by that coefficient value.

---

# Ridge Regression

Next, I trained a Ridge Regression model with

Alpha = **1.0**

### Results

| Model             |           MSE | R² Score |
| ----------------- | ------------: | -------: |
| Linear Regression | 2574508794.23 |   0.6644 |
| Ridge Regression  | 2343388667.69 |   0.6945 |

Ridge Regression performed a little better than Linear Regression.

Ridge adds a penalty to large coefficient values, so it keeps them smaller and helps reduce overfitting. Because of this, the coefficient values are a little different from Linear Regression.

The alpha value controls how strong this penalty is. A larger alpha gives stronger regularization.

---

# Logistic Regression

For classification, I trained a Logistic Regression model.

Before training, I checked the class balance.

Class 1 = **51.1%**

Class 0 = **48.9%**

Since both classes were already balanced, I did not use SMOTE or class weights.

---

# Classification Results

The model gave good results.

* Precision = **0.9014**
* Recall = **0.9771**
* AUC = **0.9708**

I also created a confusion matrix, classification report and ROC Curve.

The ROC curve showed that the model can separate both classes very well.

An AUC score of **0.9708** means the model is very good at telling the difference between higher-priced and lower-priced houses.

---

# Precision and Recall

Precision tells me how many houses that the model predicted as expensive were actually expensive.

Formula

**Precision = TP / (TP + FP)**

Recall tells me how many expensive houses were correctly found by the model.

Formula

**Recall = TP / (TP + FN)**

For this project, I think both Precision and Recall are important.

I want the model to correctly find expensive houses without making too many wrong predictions.

---

# Decision Threshold

I tested different thresholds from **0.30** to **0.70**.

| Threshold |  Precision |     Recall |   F1 Score |
| --------: | ---------: | ---------: | ---------: |
|      0.30 |     0.8889 |     0.9771 |     0.9309 |
|      0.40 |     0.8889 |     0.9771 |     0.9309 |
|  **0.50** | **0.9014** | **0.9771** | **0.9377** |
|      0.60 |     0.9130 |     0.9618 |     0.9368 |
|      0.70 |     0.9313 |     0.9313 |     0.9313 |

The best F1 Score came at **0.50**.

I would keep the threshold at **0.50** because it gives the best balance between Precision and Recall on this dataset.

If I increase the threshold, Precision becomes better but Recall goes down.

If I decrease the threshold, Recall becomes better but Precision becomes lower.

---

# Strong Regularization

Then I trained another Logistic Regression model using

**C = 0.01**

A smaller C means stronger regularization.

### Comparison

| Model    | Precision | Recall |    AUC |
| -------- | --------: | -----: | -----: |
| C = 1.0  |    0.9014 | 0.9771 | 0.9708 |
| C = 0.01 |    0.9209 | 0.9771 | 0.9858 |

The model with **C = 0.01** gave slightly better Precision and AUC.

Recall stayed the same.

The C value controls how much regularization is applied. A smaller C means stronger regularization, while a larger C means weaker regularization.

---

# Bootstrap Confidence Interval

Finally, I used bootstrap sampling **500 times** to compare both Logistic Regression models.

### Results

**Mean AUC Difference**

**-0.015681**

**95% Confidence Interval**

**-0.036591 to 0.001145**

Since this confidence interval includes **0**, I cannot confidently say that one model is always better than the other.

The difference could simply be because of random sampling.

---

# Files

This part contains:

* Part2_ML_Models.ipynb
* cleaned_data.csv
* roc_curve.png
* README.md

---

**This completes Part 2 of the project.**
