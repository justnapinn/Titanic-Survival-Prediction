# Titanic Survival Prediction

## Overview

This project aims to predict the survival of passengers aboard the Titanic based on various features such as age, gender, class, fare, and others. The goal is to develop a machine learning model that can predict whether a passenger survived or not based on their information. This project utilizes several classification algorithms and evaluates their performance to select the best-performing model.

## Problem Statement

The Titanic dataset contains information about passengers on board the Titanic, including whether they survived or not. The objective is to predict survival using features like passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, and fare.

## Dataset
The dataset used for this project is sourced from Kaggle.

You can access the dataset from the following link:
[Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

## Setup

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)

df = pd.read_csv("/root/.cache/kagglehub/datasets/yasserh/titanic-dataset/versions/1/Titanic-Dataset.csv")
```

## Steps Taken

### 1. Import Libraries  
- Imported necessary libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn` for data analysis.  
- Used `sklearn` for Machine Learning and model evaluation.  

### 2. Load Dataset  
- Loaded the dataset using `pandas.read_csv()`.  

### 3. Exploratory Data Analysis (EDA)  
- Checked for missing values using `df.isnull().sum()`.  
- Identified outliers using `boxplot`.  

### 4. Handle Missing Values  
- Used `fillna()` to handle missing data.  

### 5. Data Preprocessing  
- Applied  `pd.get_dummies()` to convert categorical features into numerical format.  

### 6. Split Dataset  
- Split the dataset into **Training Set** and **Testing Set** using `train_test_split()`.  
- Set `test_size=0.3` to allocate 70% of data for training and 30% for testing.  

### 7. Train Models  
- Built and trained multiple Machine Learning models, including:  
  - **Logistic Regression** (`LogisticRegression()`)  
  - **Decision Tree** (`DecisionTreeClassifier()`)  
  - **Random Forest** (`RandomForestClassifier()`)  
- Trained models using `model.fit(X_train, y_train)`.  

### 8. Make Predictions  
- Used `model.predict(X_test)` to make predictions on the test dataset.  

### 9. Evaluate Models  
- Calculated model performance metrics such as:  
  - `accuracy_score(y_test, y_pred)`  
  - `classification_report(y_test, y_pred)`  
  - `confusion_matrix(y_test, y_pred)`  
- Compared models based on **Accuracy, Precision, Recall, F1-score, and AUC-ROC**.  

### 10. Model Comparison  
- Created a summary table comparing model performance.  
<h2>Model Performance Metrics</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
                <th>AUC-ROC</th>
                <th>PR AUC</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Logistic Regression</td>
                <td>0.773</td>
                <td>0.467</td>
                <td>0.792</td>
                <td>0.587</td>
                <td>0.854</td>
                <td>0.663</td>
            </tr>
            <tr>
                <td>Decision Tree</td>
                <td>0.887</td>
                <td>0.724</td>
                <td>0.720</td>
                <td>0.722</td>
                <td>0.825</td>
                <td>0.751</td>
            </tr>
            <tr>
                <td>Random Forest</td>
                <td>0.931</td>
                <td>0.964</td>
                <td>0.689</td>
                <td>0.804</td>
                <td>0.924</td>
                <td>0.867</td>
            </tr>
            <tr>
                <td>XGBoost</td>
                <td>0.933</td>
                <td>0.943</td>
                <td>0.717</td>
                <td>0.815</td>
                <td>0.943</td>
                <td>0.888</td>
            </tr>
        </tbody>
    </table>

💡 **Note:**  
Model performance can be improved by fine-tuning hyperparameters using **GridSearchCV** or **RandomizedSearchCV**.  


### Conclusion
This project demonstrates the application of machine learning models to solve credit risk prediction problems. The XGBoost model was selected as the best-performing model, with high precision and recall, making it ideal for real-world credit risk applications where false positives and false negatives are critical. The code can be adapted for other financial prediction tasks with similarly structured data.
