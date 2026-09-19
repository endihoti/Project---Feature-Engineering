# Project---Feature-Engineering
# Customer Churn Prediction (Project - Feature Engineering)

## Project Overview
Beta Bank customers are leaving every month. Because acquiring new customers is significantly more expensive than retaining existing ones, the goal of this project is to build a machine learning model to predict whether a customer will terminate their contract soon.

The primary requirement for this project is achieving an **F1 score of at least 0.59** on the unseen test dataset. Additionally, the **AUC-ROC** metric is evaluated to measure overall class discrimination performance.

---

## Technical Highlights & Methodologies

* **Data Leakage Prevention:** Implemented a strict 60% Train / 20% Validation / 20% Test stratified split *before* performing any feature transformations. Imputation (median), One-Hot Encoding (`drop='first'`), and Feature Scaling (`StandardScaler`) were fitted strictly on `X_train` and applied downstream.
* **Exploratory Data Analysis (EDA):** Scoped missing values in `tenure` (909 missing rows, ~9.09% of the dataset) and justified median imputation using tenure distribution plots across churned vs. retained customers.
* **Class Imbalance Mitigation:** Handled the 4:1 class imbalance (~80% retained vs. ~20% churned) by evaluating Class Weighting (`class_weight='balanced'`), Upsampling, and Downsampling techniques across Random Forest models.
* **Threshold Tuning:** Applied probability decision threshold optimization on the validation set to maximize Recall and precision, successfully pushing the test F1 score past the project threshold.
* **Legacy Environment Safety:** Wrote version-safe preprocessing code compatible with older `scikit-learn` releases (resolving `get_feature_names_out` and `handle_unknown` deprecation errors).

---

## Dataset Overview

The dataset (`/datasets/Churn.csv`) contains historical data on client behavior and contract termination:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `CreditScore` | Quantitative | Customer's credit score |
| `Geography` | Categorical | Country of residence (France, Spain, Germany) |
| `Gender` | Categorical | Male or Female |
| `Age` | Quantitative | Customer's age in years |
| `Tenure` | Quantitative | Years as a bank client (Contains ~9.09% missing values) |
| `Balance` | Quantitative | Account balance |
| `NumOfProducts` | Quantitative | Number of banking products used |
| `HasCrCard` | Binary | Customer holds a credit card (1 = Yes, 0 = No) |
| `IsActiveMember` | Binary | Active membership status (1 = Yes, 0 = No) |
| `EstimatedSalary`| Quantitative | Estimated annual salary |
| **`Exited`** *(Target)* | Binary | Customer churn status (1 = Churned, 0 = Retained) |

*Note: Identifier features (`RowNumber`, `CustomerId`, `Surname`) were dropped prior to model training to prevent overfitting.*

---

## Key Results & Metrics

| Model Stage | Validation F1 Score | Test F1 Score | Test AUC-ROC |
| :--- | :---: | :---: | :---: |
| **Baseline Decision Tree** (Unbalanced) | 0.4871 | — | — |
| **Random Forest + Threshold Tuning** | **0.6120** | **0.6073** | **0.8542** |

> **Final Outcome:** The tuned `RandomForestClassifier` with threshold optimization achieved a **Test F1 Score of 0.6073** (exceeding the 0.59 requirement) and a **Test AUC-ROC Score of 0.8542**.

---

## Project Structure

```text
├── main.ipynb            # Clean 10-cell Jupyter notebook (Data Prep -> Modeling -> Evaluation)
├── README.md             # Project documentation and summary
└── /datasets/
    └── Churn.csv         # Bank customer churn dataset
