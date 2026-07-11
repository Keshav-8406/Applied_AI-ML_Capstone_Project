# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis 

## Dataset

In this part, I used the Ames Housing Dataset (`train.csv`).

The dataset has **1460 rows** and **81 columns**. The target column is **SalePrice**, which shows the selling price of each house.

---

# What I Did

In this part, I cleaned the dataset and explored it before building any machine learning model.

The work I completed:

- Loaded the dataset
- Checked the first 5 rows
- Checked data types and dataset size
- Found missing values
- Removed duplicate rows
- Changed some data types
- Checked statistics
- Found skewed columns
- Detected outliers
- Created different charts
- Compared Pearson and Spearman correlation
- Performed grouped analysis
- Saved the cleaned dataset

---

# Missing Values

I checked missing values in every column.

Some columns had more than **20%** missing values, so I left them because most of those missing values mean that feature was not available for that house.

For numeric columns with less than **20%** missing values, I filled the missing values using the **median**.

### Why I Used the Median

I used the median because some columns have very large values that can affect the mean.

The median gives a better middle value when the data is not evenly spread.

---

# Duplicate Rows

I checked for duplicate rows.

After removing them, I checked the missing values again to see if anything changed.

---

# Data Type Changes

I changed these columns to **category** type:

- MSSubClass
- Neighborhood

This helps reduce memory usage because these columns have many repeated values.

---

# Descriptive Statistics

I used `df.describe()` to see information like:

- Mean
- Minimum value
- Maximum value
- Standard deviation

This helped me understand the dataset better.

---

# Skewness

I checked skewness for all numeric columns.

The two most skewed columns were:

- MiscVal
- PoolArea

Both are **positively skewed**, which means most values are small and only a few values are very large.

Because of this, the median is a better choice than the mean.

---

# Outlier Detection

I used the IQR method to find outliers in:

- SalePrice
- GrLivArea

Results:

- SalePrice → 61 outliers
- GrLivArea → 31 outliers

I did not remove these outliers because they may be real house prices and not errors.

If needed, I will handle them in Part 2.

---

# Visualizations

I created these charts:

- Line Plot
- Bar Chart
- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap

### Histogram

The histogram shows that most values are close to zero, while only a few values are very high.

### Scatter Plot

The scatter plot between **GrLivArea** and **SalePrice** shows that bigger houses usually have higher prices.

### Box Plot

The box plot shows that house prices are different in different neighborhoods.

Some neighborhoods have higher house prices than others.

---

# Correlation

I calculated both Pearson and Spearman correlation.

The biggest differences were found between:

- LotFrontage and LotArea
- LotArea and BedroomAbvGr
- LotArea and TotRmsAbvGrd

The Spearman values were higher than the Pearson values, which means these columns have a strong relationship but not a perfectly straight-line relationship.

For Part 2, I will also consider Spearman correlation while selecting features.

---

# Mean vs Median

The two most skewed columns were:

- MiscVal
- PoolArea

For both columns:

- Mean was higher
- Median was 0

This shows that the mean is affected by a few very large values.

So, I used the median to fill missing values.

After filling missing values, there were no null values left in these columns.

---

# Grouped Analysis

I grouped **Neighborhood** with **SalePrice**.

Results:

- Highest Mean Group → NoRidge
- Highest Standard Deviation Group → NoRidge
- Mean Ratio → 3.4

This shows that the neighborhood has a good effect on house prices and can be useful for prediction.

---

# Output

At the end, I saved the cleaned dataset as:

**cleaned_data.csv**

This file will be used in Part 2.
