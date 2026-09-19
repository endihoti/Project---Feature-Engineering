# Customer Churn Prediction & Class Imbalance Mitigation

## Project Overview
Beta Bank is experiencing monthly customer churn. Because acquiring new clients is significantly more costly than retaining existing ones, the objective of this project is to build a high-performance binary classification model to predict whether a customer will terminate their contract soon.

To meet the project criteria, the target model must achieve an **F1 score $\ge$ 0.59** on the unseen test dataset. Additionally, the **AUC-ROC** metric is evaluated across model iterations to gauge overall class discrimination capability.

---

## Technical Highlights & Evaluation Criteria

* **Data Preprocessing & Data Leakage Prevention:**
  * Dropped irrelevant identifier features (`RowNumber`, `CustomerId`, `Surname`) to prevent model overfitting.
  * Extracted features and target (`Exited`), followed by a strict **60% Train / 20% Validation / 20% Test** stratified split executed *before* any feature transformations.
  * Imputed missing values in `Tenure` (~9.09% missing) using training set medians, performed One-Hot Encoding (`drop='first'`) on categorical variables (`Geography`, `Gender`), and scaled numeric features via `StandardScaler` fitted strictly on `X_train`.
* **Class Imbalance Investigation:**
  * Identified a ~4:1 class imbalance (~80% retained vs. ~20% churned).
  * Evaluated baseline models (Decision Tree & Random Forest) without handling class imbalance to establish benchmark performance.
* **Imbalance Handling & Tuning (2+ Approaches):**
  * Evaluated **Class Weighting** (`class_weight='balanced'`).
  * Implemented **Upsampling** (oversampling the minority churn class) and **Downsampling**.
  * Optimized decision probability thresholds on the validation set to maximize precision and recall balance.
* **Final Model Selection & Testing:**
  * Hyperparameter tuning across tree depth and estimators identified the tuned `RandomForestClassifier` as the optimal model.
  * Final evaluation conducted on the isolated test set to confirm threshold criteria compliance.

---

## Dataset Overview

Dataset source: `/datasets/Churn.csv`

| Feature | Type | Description |
| :--- | :--- | :--- |
| `CreditScore` | Quantitative | Customer credit score |
| `Geography` | Categorical | Country of residence (France, Spain, Germany) |
| `Gender` | Categorical | Gender |
| `Age` | Quantitative | Customer age in years |
| `Tenure` | Quantitative | Years as a bank client (Contains ~9.09% missing values) |
| `Balance` | Quantitative | Account balance |
| `NumOfProducts` | Quantitative | Number of banking products used |
| `HasCrCard` | Binary | Credit card holder status (1 = Yes, 0 = No) |
| `IsActiveMember` | Binary | Active membership status (1 = Yes, 0 = No) |
| `EstimatedSalary` | Quantitative | Estimated annual salary |
| **`Exited`** *(Target)* | Binary | Customer churn status (1 = Churned, 0 = Retained) |

---

## Key Model Results & Metrics

| Model Approach | Validation F1 Score | Test F1 Score | Test AUC-ROC |
| :--- | :---: | :---: | :---: |
| **Baseline Decision Tree** (Unbalanced) | 0.4871 | — | — |
| **Random Forest** (Balanced / Upsampled) | 0.5890 | — | — |
| **Random Forest + Threshold Optimization** | **0.6120** | **0.6073** | **0.8542** |

> **Final Outcome:** The tuned Random Forest model achieved a **Test F1 Score of 0.6073** (exceeding the 0.59 project requirement) and an **AUC-ROC Score of 0.8542**, demonstrating robust class separation and predictive performance on unseen data.

---

## Project Structure

```text
├── main.ipynb            # Jupyter notebook with complete data prep, imbalance experiments, and test evaluation
├── README.md             # Project documentation and summary
└── /datasets/
    └── Churn.csv         # Beta Bank customer dataset
