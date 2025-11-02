# 🏡 House Price Prediction Using Machine Learning: A Complete Technical Tutorial

**Author:** Soma Samanta  
**Category:** Machine Learning, Data Science, Streamlit  
**Estimated Read Time:** 8 minutes

---

## 📖 Introduction

Predicting house prices has always been one of the most exciting applications of machine learning. Whether you’re a data science beginner or a real estate professional, understanding how machine learning models estimate property values can provide deep insights into the factors that influence prices.

In this tutorial, we’ll go step-by-step through a **complete data science workflow** for predicting house prices using **Exploratory Data Analysis (EDA)**, **feature engineering**, and **machine learning modeling**. By the end, you’ll not only have a predictive model but also understand the reasoning behind each step.

---

## 🧩 Step 1: Understanding the Dataset

We’ll be using the popular **House Prices: Advanced Regression Techniques** dataset from **Kaggle**.

The dataset contains **79 explanatory variables** describing almost every aspect of residential homes in Ames, Iowa — such as the number of rooms, type of foundation, roof material, and quality of construction.

### 📁 Files

- `train.csv`: Contains 1460 observations with the target variable (`SalePrice`).
- `test.csv`: Contains 1459 observations without `SalePrice`, used for prediction.

### 🏠 Key Columns

| Feature | Description |
|----------|--------------|
| `LotArea` | Lot size in square feet |
| `OverallQual` | Overall material and finish quality (1–10) |
| `YearBuilt` | Original construction date |
| `GrLivArea` | Above ground living area in square feet |
| `TotalBsmtSF` | Total basement area |
| `GarageCars` | Size of garage in car capacity |
| `SalePrice` | Property sale price (target) |

---

## 🔍 Step 2: Exploratory Data Analysis (EDA)

EDA is the most critical part of any machine learning project. It helps uncover patterns, relationships, and anomalies.

### 2.1. Data Overview

We start by checking the shape, data types, and missing values:

```python
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.info()
train.describe()
train.isnull().sum().sort_values(ascending=False).head(20)
```

### 2.2. Target Variable Distribution

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train['SalePrice'], kde=True)
plt.title('Sale Price Distribution')
plt.show()
```

**Observation:** The `SalePrice` distribution is right-skewed, suggesting we may need a log transformation.

### 2.3. Correlation Heatmap

```python
corr = train.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()
```

**Key Insights:**
- `OverallQual`, `GrLivArea`, and `GarageCars` are highly correlated with `SalePrice`.
- Some features like `PoolArea` and `MiscVal` have almost no impact.

### 2.4. Missing Value Imputation

Categorical missing values are replaced with `'None'`, and numerical ones with `0` or the median.

```python
train.fillna({'PoolQC':'None','GarageType':'None'}, inplace=True)
train.fillna(train.median(numeric_only=True), inplace=True)
```

---

## ⚙️ Step 3: Feature Engineering

We create new features to improve predictive power.

```python
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train['Age'] = train['YrSold'] - train['YearBuilt']
```

Next, we encode categorical variables using **One-Hot Encoding** or **Ordinal Encoding**.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_features = train.select_dtypes(include=['object']).columns
num_features = train.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])
```

---

## 🧠 Step 4: Model Building

We’ll use **Random Forest Regressor**, a robust algorithm for tabular data.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

X = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([('preprocessor', preprocessor),
                  ('rf', RandomForestRegressor(random_state=42))])

model.fit(X_train, y_train)
score = model.score(X_valid, y_valid)
print(f'Validation R²: {score:.2f}')
```

---

## 📊 Step 5: Model Evaluation

We use **R² Score** and **RMSE** for evaluation.

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_valid)
r2 = r2_score(y_valid, y_pred)
rmse = mean_squared_error(y_valid, y_pred, squared=False)

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")
```

---

## 🌐 Step 6: Streamlit Web App

Once satisfied with the model, we deploy it using **Streamlit** for real-time predictions.

```python
import streamlit as st
import joblib

model = joblib.load('models/final_model_pipeline.joblib')
st.title('🏡 House Price Prediction App')

lot_area = st.number_input('Lot Area', value=8000)
year_built = st.number_input('Year Built', value=2000)
total_sf = st.number_input('Total SF', value=2500)

if st.button('Predict'):
    X = pd.DataFrame([{'LotArea': lot_area, 'YearBuilt': year_built, 'TotalSF': total_sf}])
    prediction = model.predict(X)[0]
    st.success(f"Predicted Price: ${prediction:,.0f}")
```

To run it:

```bash
streamlit run app.py
```

---

## 🚀 Conclusion

We successfully built a complete **House Price Prediction System** — from **EDA** to **deployment**.

**What we learned:**
- How to clean and preprocess real-world datasets  
- How to visualize data insights with EDA  
- How to build and evaluate machine learning models  
- How to deploy your model using Streamlit  

This project demonstrates the **end-to-end ML lifecycle**, helping you strengthen your technical portfolio.

---

**📌 Author:** *Soma Samanta*  
**🗓️ Date:** November 2025  
**🔗 Follow me for more projects in Machine Learning, Deep Learning, and Generative AI!*
