# 📉 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20GridSearchCV-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-189FDD)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Predicting which telecom customers are likely to churn, using their demographic profile, subscribed services, contract type and billing history.

**Result:** XGBoost with ROC-AUC **0.83** on validation and the smallest train/validation gap (0.039), plus business recommendations based on churn drivers.

## 📋 Table of Contents

* [🧭 About the Project](#-about-the-project)
* [🗂 Project Structure](#-project-structure)
* [🔬 Methodology](#-methodology)
* [⚙️ Getting Started](#️-getting-started)
* [📊 Results](#-results)
* [💡 Business Insights](#-business-insights)
* [📝 License](#-license)

## 🧭 About the Project

Customer churn — when a client stops using a company's services — is one of the most expensive problems for subscription-based businesses. Acquiring a new customer typically costs far more than retaining an existing one, which makes **early churn detection** a high-value use case for machine learning.

This project builds an end-to-end churn prediction pipeline on the classic **Telco Customer Churn** dataset (7,043 customers), covering:

* 🧹 Data cleaning and preprocessing
* 📊 Exploratory data analysis (EDA) with visualizations
* ⚖️ Handling class imbalance
* 🎛 Hyperparameter tuning with cross-validation
* 🧪 Model evaluation using imbalance-aware metrics (ROC-AUC, PR-AUC)

The final goal is to flag at-risk customers so that a retention team can proactively intervene (special offers, support outreach, contract upgrades, etc.).

## 🗂 Project Structure

```
customer-churn-prediction/
├── Customer_Churn_Prediction_.ipynb   # Main notebook: EDA + modeling
├── requirements.txt                   # Python dependencies
├── LICENSE
└── README.md
```

> The dataset is not stored in the repository — see [step 4](#4-add-the-dataset) below.

### Notebook walkthrough
| Section | What happens |
| :--- | :--- |
| **1. Data Loading & Cleaning** | Load the raw CSV, inspect shape/dtypes, fix `TotalCharges` (stored as text with blanks → coerced to numeric), drop the `customerID` identifier, check for duplicates/nulls |
| **2. Feature Encoding** | Map binary Yes/No columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) to 0/1; normalize service columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `MultipleLines`) by collapsing `"No internet/phone service"` into `"No"` |
| **3. Exploratory Data Analysis** | Boxplots for `MonthlyCharges` / `TotalCharges`, churn rate breakdown by payment method & contract type, tenure distribution (KDE) split by churn, full numerical correlation heatmap |
| **4. Train/Validation Split** | 80/20 split with `train_test_split` (`random_state=0`) |
| **5. Preprocessing Pipeline** | `ColumnTransformer` combining `OneHotEncoder` for low-cardinality categorical features and a `SimpleImputer` for numerical features, wrapped in an sklearn `Pipeline` |
| **6. Modeling — Random Forest** | `RandomForestClassifier` tuned via `GridSearchCV` (5-fold CV, `roc_auc` scoring) |
| **7. Modeling — XGBoost** | `XGBClassifier` tuned via `GridSearchCV`, with `scale_pos_weight` computed from the class ratio to counter imbalance |
| **8. Evaluation** | Custom `evaluate_model()` helper reporting Train/Validation ROC-AUC, PR-AUC (Average Precision) and a full `classification_report` |

## 🔬 Methodology

### ⚖️ Handling Class Imbalance

The dataset is imbalanced — roughly **73% retained vs. 27% churned** customers — so accuracy alone is a misleading metric. Two complementary strategies were used:

* **Random Forest** → `class_weight='balanced'` (tuned as a grid search option, alongside `None`)
* **XGBoost** → `scale_pos_weight` computed as the ratio of negative to positive class counts (`neg / pos ≈ 2.75` in the training set), also tuned in the grid

### 🎛 Hyperparameter Tuning

Both models were optimized with **`GridSearchCV`** using **5-fold cross-validation**, optimizing directly for **ROC-AUC**:

```python
rf_param_grid = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [5, 10, 15],
    'model__min_samples_leaf': [1, 5, 10],
    'model__max_features': ['sqrt', 'log2'],
    'model__class_weight': ['balanced', None],
}

xgb_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0],
    'model__reg_lambda': [1, 5],
    'model__scale_pos_weight': [1, scale_pos_weight],
}
```

### 📏 Evaluation Metrics

Since churn is a minority-class problem, models were compared using:

* **ROC-AUC** — overall ability to rank churners above non-churners
* **PR-AUC (Average Precision)** — more informative than ROC-AUC on imbalanced data, since it focuses on the positive (churn) class
* **Precision / Recall / F1** per class, via `classification_report`
* **Train vs. Validation gap** — used as a proxy for overfitting

## ⚙️ Getting Started

### Prerequisites

* Python 3.10+
* Jupyter Notebook / JupyterLab (or Google Colab)

### 1. Clone the repository

```bash
git clone https://github.com/ValeriaZav26/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the **Telco Customer Churn** dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root (or update the path in the notebook).

### 5. Run the notebook

```bash
jupyter notebook Customer_Churn_Prediction_.ipynb
```

## 📊 Results

Both models were evaluated on the held-out validation set (20% of the data, 1,409 customers):

| Model | Train ROC-AUC | Valid ROC-AUC | Valid PR-AUC | Overfit Gap |
| :--- | :---: | :---: | :---: | :---: |
| 🌲 Random Forest | 0.9015 | 0.8315 | 0.6344 | 0.070 |
| 🚀 XGBoost | 0.8674 | **0.8288** | 0.6326 | **0.039** |

> **🏆 Selected model: XGBoost.** Both models reach a very similar validation ROC-AUC (~0.83), but XGBoost generalizes noticeably better — its train/validation gap is about **half** that of Random Forest (0.039 vs. 0.070), indicating less overfitting.
>
> ⚠️ **Caveat:** Recall on the churn class is still moderate (~0.48) — the model currently misses more than half of the customers who actually churn. For a real retention campaign, it's worth **lowering the classification threshold** below 0.5 to trade some precision for higher recall, since the cost of missing a churner is usually higher than the cost of a false alarm.

## 💡 Business Insights

* **Contract type is the strongest churn driver:** month-to-month customers churn at **42.7%** vs. **2.8%** for two-year contracts.
* **Payment method matters:** customers paying by electronic check churn at **45.3%** vs. 15–19% for other methods.
* **Recommendation:** offer incentives to move customers to annual contracts and automatic payments.

## 📝 License

This project is licensed under the [MIT License](LICENSE).
