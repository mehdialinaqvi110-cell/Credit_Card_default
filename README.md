# Credit Card Default Prediction

A simple Machine Learning project that predicts whether a credit card customer is likely to **default on their payment** based on financial and repayment-related information.

The project uses **Support Vector Machine (SVM)** for classification and a simple **Streamlit application** for making predictions.

## Project Overview

Credit card default prediction is a classification problem where the objective is to identify whether a customer is likely to default based on their financial and repayment history.

In this project, I performed:

* Data loading and exploration
* Exploratory Data Analysis (EDA)
* Feature engineering
* Feature selection
* Data preprocessing
* SVM model training
* Model evaluation
* Model saving using Pickle
* Simple Streamlit deployment

## Dataset

The dataset contains customer information related to:

* Credit limit
* Age
* Education
* Repayment status
* Bill amounts
* Payment amounts
* Other customer financial information

### Target Variable

`default`

* `Y` → Customer defaults
* `N` → Customer does not default

## Exploratory Data Analysis

During EDA, I analyzed the relationship between different financial variables and credit card default.

Some of the analysis included:

* Credit limit vs repayment status
* Age vs credit limit
* Bill amount vs payment amount
* Payment-to-bill ratio
* Repayment history and default

One feature engineered during EDA was:

**Payment-to-Bill Ratio**

```python
PAYMENT_RATIO = PAY_AMT1 / (abs(BILL_AMT1) + 1)
```

The analysis showed that individual bill and payment variables have considerable overlap between defaulters and non-defaulters, so multiple features were used together for prediction.

## Machine Learning Approach

### 1. Feature Selection

After preprocessing and feature analysis, the following features were selected for the final model:

```text
LIMIT_BAL
PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6
PAY_AMT1
PAY_AMT2
PAY_AMT3
PAY_AMT4
PAY_AMT5
PAY_AMT6
DELAY_COUNT
EDUCATION
```

### 2. Data Preprocessing

The project uses a `ColumnTransformer` to preprocess the data.

**Numerical features:**

* RobustScaler

**Categorical features:**

* OneHotEncoder

RobustScaler was used to reduce the effect of extreme values in financial variables.

### 3. Train-Test Split

The dataset was divided into:

* 80% training data
* 20% testing data

Stratified splitting was used to maintain the class distribution.

```python
train_test_split(
    x_best,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Model

### Support Vector Machine (SVM)

The final model uses `SVC` from Scikit-learn.

The preprocessing and model were combined into a single Pipeline:

```python
model_pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", SVC())
])
```

This makes it easier to apply the same preprocessing when making predictions on new customer data.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

## Streamlit Application

A simple **Streamlit application** was created to provide an easy interface for making predictions.

The user enters the required customer information, and the application uses the trained SVM model to predict whether the customer is likely to default.

The Streamlit application is intentionally simple and focuses on the basic prediction workflow rather than a complex dashboard.

### Application Flow

```text
User Input
    ↓
Streamlit Interface
    ↓
Saved SVM Model
    ↓
Preprocessing
    ↓
Prediction
    ↓
Default / No Default
```

## Project Structure

```text
Credit-Card-Default-Prediction/
│
├── Credit Card Defaulter Prediction.csv
├── Credit_Card_Default.ipynb
├── credit_card_default.pkl
├── app.py
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Pickle

## Machine Learning Concepts Used

* Exploratory Data Analysis
* Feature Engineering
* Feature Selection
* Train-Test Split
* Data Preprocessing
* Robust Scaling
* One-Hot Encoding
* Support Vector Machine
* Classification
* Model Evaluation
* Machine Learning Pipeline

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Default-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Credit-Card-Default-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

## Key Learning Outcomes

Through this project, I learned how to:

* Perform EDA on a real-world classification dataset
* Identify useful features for prediction
* Handle numerical and categorical features
* Apply feature scaling
* Use feature selection techniques
* Build an SVM classification model
* Create a preprocessing and ML pipeline
* Evaluate a classification model
* Save a trained model using Pickle
* Build a simple Streamlit ML application

## Disclaimer

This project is developed for **educational and demonstration purposes**. The prediction should not be used as the sole basis for real-world credit decisions.
