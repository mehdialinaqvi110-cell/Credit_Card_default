# 💳 Credit Card Default Prediction

## 📌 Project Overview

This project focuses on predicting whether a credit card customer is likely to **default on their payment** based on demographic information, credit limit, repayment history, bill amounts, and payment amounts.

The project follows an end-to-end **Machine Learning workflow**, including data exploration, feature engineering, preprocessing, feature selection, model training, evaluation, and model serialization.

The dataset contains **30,000 customer records and 25 columns**. The target variable is `default`, which indicates whether the customer defaulted on their credit card payment.

---

## 🎯 Objective

The main objective of this project is to:

* Analyze customer financial and demographic characteristics.
* Identify factors associated with credit card default.
* Perform exploratory data analysis (EDA).
* Engineer useful features from repayment history.
* Select the most relevant features.
* Build a machine learning classification model.
* Evaluate the model using classification metrics.
* Save the trained model for deployment.

---

## 📊 Dataset

The dataset contains **30,000 observations and 25 columns**. It includes demographic, credit, repayment, billing, and payment information.

### Main Features

| Feature                   | Description                           |
| ------------------------- | ------------------------------------- |
| `LIMIT_BAL`               | Credit limit assigned to the customer |
| `SEX`                     | Customer gender                       |
| `EDUCATION`               | Education level                       |
| `MARRIAGE`                | Marital status                        |
| `AGE`                     | Customer age                          |
| `PAY_0`                   | Most recent repayment status          |
| `PAY_2` - `PAY_6`         | Previous repayment statuses           |
| `BILL_AMT1` - `BILL_AMT6` | Previous billing amounts              |
| `PAY_AMT1` - `PAY_AMT6`   | Previous payment amounts              |
| `default`                 | Target variable indicating default    |

The `ID` column was removed because it is an identifier and does not provide useful predictive information.

---

## 🔍 Exploratory Data Analysis

Several EDA techniques were performed to understand the dataset and identify relationships with credit card default.

### EDA Performed

* Dataset structure and data types
* Missing-value analysis
* Duplicate-value analysis
* Distribution analysis
* Boxplots for numerical variables
* Default distribution
* Credit limit vs default
* Age group vs default
* Gender vs default
* Education vs default
* Marital status vs default
* Repayment status vs default
* Bill amount vs default
* Payment amount vs default
* Credit limit and repayment status analysis

### Key Findings

#### 💰 Credit Limit

Credit limit was analyzed against default behavior to understand whether credit exposure is associated with repayment risk.

#### 👤 Age

Customers were grouped into age categories:

* 21–30
* 31–40
* 41–50
* 51–60
* 61+

The analysis showed that customers in the **21–40 age groups represented a notable portion of defaults**.

#### 🎓 Education

University-educated customers represented a large portion of the default observations in the dataset.

#### 💍 Marital Status

The analysis indicated that **married customers had more defaults than single customers**.

#### 💳 Repayment Status

Repayment status showed one of the strongest relationships with default. Customers with delayed repayments had a substantially higher likelihood of default.

---

## 🛠️ Feature Engineering

A new feature called `DELAY_COUNT` was created from the repayment-status columns:

```python
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

df['DELAY_COUNT'] = (df[pay_cols] > 0).sum(axis=1)
```

This feature represents the **number of months in which the customer had a positive repayment delay**.

The analysis showed that:

> As the number of payment delays increases, the default rate also increases.

Another feature, `PAYMENT_RATIO`, was also explored:

```python
df['PAYMENT_RATIO'] = (
    df['PAY_AMT1'] / (df['BILL_AMT1'].abs() + 1)
)
```

This represents the relationship between payment amount and bill amount.

---

## ⚙️ Data Preprocessing

The dataset was divided into training and testing sets using an **80:20 split** with stratification:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Numerical Features

Numerical variables were scaled using:

**RobustScaler**

Robust scaling was selected to make the preprocessing less sensitive to extreme values.

### Categorical Features

Categorical variables were transformed using:

**OneHotEncoder**

```python
OneHotEncoder(handle_unknown='ignore')
```

A `ColumnTransformer` was used to apply the appropriate preprocessing to numerical and categorical columns.

---

## 🎯 Feature Selection

To reduce the number of features and retain the most informative variables, **SelectKBest with ANOVA F-test (`f_classif`)** was used.

The top **8 features** selected by the model were:

```text
LIMIT_BAL
PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6
DELAY_COUNT
```

These features were selected based on their statistical relationship with the target variable.

---

## 🤖 Machine Learning Model

### Logistic Regression

A **Logistic Regression** classifier was trained using the selected features.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2')
lr.fit(best_features, y_train)
```

Logistic Regression was used as the classification model to predict whether a customer would default.

---

## 📈 Model Evaluation

The model was evaluated on the test dataset using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

### Accuracy

The model achieved approximately:

**🎯 Accuracy: 80.9%**

The classification report showed stronger performance for the non-default class, while identifying default customers was more challenging.

| Class             | Precision | Recall | F1-Score |
| ----------------- | --------: | -----: | -------: |
| Non-Default (`N`) |      0.96 |   0.82 |     0.89 |
| Default (`Y`)     |      0.28 |   0.66 |     0.40 |

This indicates that the model achieved a relatively high overall accuracy while maintaining a **66% recall for default customers**, which is an important metric for identifying potential defaulters.

---

## 💾 Model Saving

The trained Logistic Regression model was saved using Python's `pickle` module:

```python
import pickle

with open("credit_card_default.pkl", "wb") as f:
    pickle.dump(lr, f)
```

The saved model can later be loaded into a deployment application such as **Streamlit**.

---

## 📁 Project Structure

```text
Credit-Card-Default-Prediction/
│
├── Credit_Card_Default.ipynb
├── Credit Card Defaulter Prediction.csv
├── credit_card_default.pkl
├── app.py
├── requirements.txt
└── README.md
```

> The exact files in your repository may vary depending on which files you upload to GitHub.

---

## 🖥️ Streamlit Deployment

The trained model can be integrated into a Streamlit application to allow users to enter customer information and receive a default prediction.

Example workflow:

```text
User Input
    ↓
Data Preprocessing
    ↓
Selected Features
    ↓
Logistic Regression Model
    ↓
Prediction
    ↓
Default / Non-Default
```

---

## 🧰 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **SciPy**
* **Scikit-learn**
* **Pickle**
* **Streamlit**
* **Google Colab / Jupyter Notebook**

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Robust Scaling + One-Hot Encoding
   ↓
SelectKBest Feature Selection
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to identify customers who are potentially at risk of credit card default.

The analysis highlights the importance of **repayment behavior**, particularly delayed payments, in predicting default. The final Logistic Regression model achieved **80.9% test accuracy**, with a **66% recall for the default class**.

The project also demonstrates an end-to-end machine learning workflow from **EDA and feature engineering to model development and deployment readiness**.

---

## 👨‍💻 Author

**Mehdi Ali**

B.Tech – Biotechnology, NIT Warangal

### Skills Demonstrated

`Python` • `Pandas` • `NumPy` • `EDA` • `Machine Learning` • `Scikit-learn` • `Feature Engineering` • `Feature Selection` • `Data Visualization` • `Streamlit`

---

⭐ If you find this project useful, consider giving the repository a star!
