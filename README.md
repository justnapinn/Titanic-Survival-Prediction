# Titanic Survival Prediction

## Overview

This project aims to predict the survival of passengers aboard the Titanic based on various features such as age, gender, class, fare, and others. The goal is to develop a machine learning model that can predict whether a passenger survived or not based on their information. This project utilizes several classification algorithms and evaluates their performance to select the best-performing model.

## Problem Statement

The Titanic dataset contains information about passengers on board the Titanic, including whether they survived or not. The objective is to predict survival using features like passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, and fare.

## Setup

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)

df = pd.read_csv("/root/.cache/kagglehub/datasets/yasserh/titanic-dataset/versions/1/Titanic-Dataset.csv")
```

## Steps Taken

### 1. Data Preprocessing:
- Loaded the Titanic dataset and performed initial cleaning.
- Dropped unnecessary columns (`Ticket`, `Cabin`) and handled missing values (imputed missing values for `Age`).
- Selected important features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

### 2. Outlier Removal:
- Detected and removed outliers in features like `Age`, `Fare`, `SibSp`, and `Parch` using the **IQR method** (Interquartile Range).
  - Calculated the **IQR** for each feature.
  - Defined outliers as values below the lower bound or above the upper bound of the IQR.
  - Removed rows with outliers to ensure cleaner data for model training.

#### Example Code for Outlier Removal:
```python
Q1_Age = df['Age'].quantile(0.25)
Q3_Age = df['Age'].quantile(0.75)
IQR_Age = Q3_Age - Q1_Age
lower_bound_Age = Q1_Age - 1.5 * IQR_Age
upper_bound_Age = Q3_Age + 1.5 * IQR_Age
df_filtered = df[(df['Age'] >= lower_bound_Age) & (df['Age'] <= upper_bound_Age)]
```
### 3. Feature Engineering:
- Converted categorical features (Sex, Embarked) into numeric values using one-hot encoding (pd.get_dummies).
- Created a new feature Female for gender encoding (0 = male, 1 = female).

### 4. Model Training and Evaluation:
- Split the data into training and testing sets using train_test_split.
- Evaluated four different classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
 
### 5. Best Model Selection:
- Compared the accuracy of all models and selected XGBoost as the best-performing model with an accuracy of 82.18%.

### Conclusion
This project demonstrates the application of machine learning models to solve classification problems. The XGBoost model showed the best performance in predicting Titanic survival, with a testing accuracy of 82.18%. The code can be adapted for other classification tasks with similarly structured data.
