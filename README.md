# ML1 — Rental Listing Inquiries

[![Open in nbviewer](https://img.shields.io/badge/Open%20notebook-nbviewer-orange)](https://nbviewer.org/github/klarness/ML1_Rental_Listing_Inquiries/blob/main/src/ml_intro.ipynb)

## About

Introductory ML project based on a [Kaggle competition](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data) from Two Sigma. The goal is to predict the price of a rental apartment listing using listing features such as number of bedrooms, bathrooms, and interest level.

**Dataset:** `train.json` from renthop.com — a real-world apartment listing dataset.

## What was explored

- Exploratory data analysis: distributions, outliers, percentiles, correlation
- Statistical analysis: histograms, boxplots, kernel density
- Feature engineering with `PolynomialFeatures`
- Model training and comparison: Linear Regression, Decision Tree, Naive (mean/median) baseline
- Evaluation metrics: MAE and RMSE on train/test splits

## Tasks

1. **Introduction**
   1. 5 real-life examples of ML applications and their benefits
   2. Classify the examples as supervised/unsupervised tasks
   3. Difference between multiclass and multilabel classification
   4. Is house price prediction a classification or regression problem? Can regression be reduced to classification?

2. **Data Analysis**
   1. Import libraries: pandas, numpy, sklearn, lightgbm, scipy, statsmodels, matplotlib, seaborn
   2. Load `train.json` with pandas
   3. Dataset shape (rows and columns)
   4. List columns; identify the target column
   5. Run `info()`, `describe()`, `corr()` — explain results, check for nulls
   6. Keep only: `bathrooms`, `bedrooms`, `interest_level`, `price`

3. **Statistical Analysis**
   1. Target analysis
      1. Histogram of target distribution
      2. Boxplot — identify outliers
      3. Drop rows outside 1st and 99th percentiles
      4. Plot histogram again and explain the change
   2. Feature analysis
      1. Type of `interest_level` column
      2. Value counts for `interest_level`
      3. Encode `interest_level` as numeric (0, 1, 2)
      4. Histograms for `bathrooms` and `bedrooms`
   3. Complex analysis
      1. Correlation matrix + heatmap
      2. Scatterplots: target (X) vs each feature (Y)

4. **Feature Engineering**
   1. Create squared features; plot updated correlation matrix
   2. Split data into train/test
   3. Apply `PolynomialFeatures(degree=10)` to `bathrooms` and `bedrooms`

5. **Model Training**
   1. Results tables: `result_MAE` and `result_RMSE` (columns: `model`, `train`, `test`)
   2. Linear Regression — fit, predict, compute MAE & RMSE
   3. Decision Tree (`random_state=21`) — fit, predict, compute MAE & RMSE
   4. Naive models — mean and median baselines, compute MAE & RMSE
   5. Compare all models; identify the best one
   6. *(Optional)* Use the full dataset with all features
