# 🌍 Financial Inclusion in Africa -- Machine Learning Project

> **Predicting bank account ownership across East Africa using a robust,
> bias-aware machine learning pipeline.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)



## 📌 Project Overview

Financial inclusion is a key driver of economic development.\
This project builds an **end-to-end machine learning system** to predict
whether an individual is likely to **own or use a bank account**, using
demographic and socio-economic survey data from East Africa.

The dataset originates from the **Zindi -- Financial Inclusion in Africa
Challenge** and contains information on approximately **33,600
individuals**.


## 🎯 Objective

-   Build a **fair and unbiased classifier**
-   Handle **class imbalance** correctly
-   Prevent **data leakage and feature mismatch**
-   Deploy a **reproducible Streamlit application**

------------------------------------------------------------------------

## 🗂️ Project Structure

    project/
    ├── data/
    │   └── Financial_inclusion_dataset.csv
    ├── notebooks/
    │   └── EDA.ipynb
    ├── scripts/
    │   ├── preprocess.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── validate.py
    │   └── main.py
    ├── models/
    │   ├── bank_account_pipeline.pkl
    │   └── target_encoder.pkl
    ├── app/
    │   └── app.py
    ├── requirements.txt
    └── README.md



## 📊 Dataset Description

### Target Variable

  Column         Description
  -------------- -------------------------------------------------------
  bank_account   Whether the respondent owns a bank account (Yes / No)

### Feature Categories

-   Demographics: age, gender, household size
-   Socio-economic: education level, job type
-   Geography: country, location type
-   Access indicators: cellphone access

------------------------------------------------------------------------

## 📒 Exploratory Data Analysis (EDA)

EDA was conducted **only in notebooks**, not in production code.

Key findings: - Severe **class imbalance** - Strong correlation between
education level and bank account ownership - Country-level differences
in inclusion rates

These insights informed **model design decisions**.

------------------------------------------------------------------------

## ⚠️ What Went Wrong & What I Fixed

### ❌ Problem 1: Education Bias

**Issue** - Early models predicted "Yes" almost exclusively for tertiary
education.

**Root Cause** - `LabelEncoder` was incorrectly used on categorical
features, introducing ordinal bias.

**Fix** - Replaced with `OneHotEncoder` inside a `ColumnTransformer`.

------------------------------------------------------------------------

### ❌ Problem 2: Class Imbalance

**Issue** - Model favored majority class (No bank account).

**Fix** - Used `class_weight="balanced"` - Stratified train-test split -
Evaluated using ROC-AUC instead of accuracy

------------------------------------------------------------------------

### ❌ Problem 3: Deployment Feature Mismatch

**Issue** - Streamlit app failed due to missing columns (`uniqueid`).

**Root Cause** - Identifier columns were included during training.

**Fix** - Removed non-informative ID columns at training time - Enforced
schema consistency via pipeline

------------------------------------------------------------------------

### ❌ Problem 4: Pipeline Failures

**Issue** - Custom transformers broke sklearn compatibility.

**Fix** - Refactored into a single sklearn-compliant pipeline -
Eliminated custom wrappers

------------------------------------------------------------------------

## 🧠 Modeling Approach

-   **Algorithm**: Logistic Regression
-   **Why Logistic Regression?**
    -   Interpretability
    -   Robustness on tabular data
    -   Deployment-friendly

### Preprocessing Pipeline

  Feature Type   Method
  -------------- ----------------
  Numerical      StandardScaler
  Categorical    OneHotEncoder
  Target         LabelEncoder

All steps are encapsulated in a **single pipeline**.

------------------------------------------------------------------------

## 📈 Model Performance

### Evaluation Metrics (Test Set)

  Metric            Score
  ----------------- --------
  Accuracy          \~79%
  Precision (Yes)   \~0.38
  Recall (Yes)      \~0.76
  F1-score (Yes)    \~0.51
  ROC-AUC           \~0.86

> ROC-AUC was prioritized due to class imbalance.

------------------------------------------------------------------------

## 🚀 Streamlit Application

Features: - User-friendly form inputs - Real-time probability
predictions - Threshold-based classification - Ethical disclaimer

### Run Locally

``` bash
streamlit run app/app.py
```


## 🛠️ How to Run the Project

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Train & evaluate model

``` bash
python scripts/main.py
```


## 🧪 Key Learnings

-   Improper encoding introduces silent bias
-   Pipelines prevent training--inference mismatch
-   Probability ≠ decision
-   EDA insights must inform modeling decisions


## 🏁 Conclusion

This project demonstrates a **production-ready machine learning
workflow**: - Bias-aware preprocessing - Robust pipeline design -
Transparent evaluation - Safe deployment

It reflects **real-world ML engineering practices**, not just model
training.


## 👤 Author

**Bright Joses**\
Data Scientist \| Machine Learning Engineer
