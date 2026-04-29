** Loan Default Prediction — EDA & ML Pipeline

-> Project Overview
End-to-end machine learning project on a loan default dataset (148,670 rows, 34 features).  
Covers exploratory data analysis, data leakage investigation, and a full sklearn pipeline.

---

-> Results
| Model                          | ROC-AUC | PR-AUC |
|--------------------------------|---------|--------|
| XGBoost (final)                | 0.876   | 0.800  |
| Logistic Regression (baseline) | 0.754   | 0.554  |

* Initial model scored ROC-AUC = 1.0 due to data leakage — identified and resolved.

---

-> What was done

-> Exploratory Data Analysis
- Classified missing values by type (MCAR / MAR / MNAR) and proposed handling strategies
- Analyzed feature distributions using skewness and kurtosis metrics
- Detected outliers via box plots and IQR analysis
- Applied statistical tests to assess feature relationships and target associations:
  - Chi-square, Cramér's V — categorical feature correlations
  - Information Value (IV) / WoE — feature predictive power
  - Variance Inflation Factor (VIF) — multicollinearity detection
  - Mann-Whitney U — numerical feature-to-target association

-> ML Pipeline
- Built full `sklearn` Pipeline with `ColumnTransformer` for parallel preprocessing paths
- Imputation strategy per missing data type (median/mode fill + missingness flags for MNAR)
- Yeo-Johnson transformation on skewed numerical features
- Rare label grouping for low-frequency categories (< 1%)
- Ordinal encoding for ordered categories, One-Hot encoding for nominal categories
- StandardScaler / RobustScaler assigned per column based on residual outlier analysis

-> Data Leakage Investigation
Identified **6 post-default recorded columns** causing perfect target separation:

| Column                 | Missing % for Defaulters            | Decision |
|------------------------|-------------------------------------|----------|
| `rate_of_interest`     | 99.5%                               | Dropped  |
| `upfront_charges`      | 99.6%                               | Dropped  |
| `property_value`       | 40.9%                               | Dropped  |
| `interest_rate_spread` | Structural issue                    | Dropped  |
| `credit_type`          | EQUI category = 99.99% default rate | Dropped  |
| `ltv`                  | Derived feature, high VIF           | Dropped  |

Removing these reduced ROC-AUC from **1.0 → 0.876** — a honest, generalizable result.

---

-> Stack
`Python` `scikit-learn` `XGBoost` `pandas` `numpy` `matplotlib` `seaborn`

---

-> Files
| File                            | Description          |
|---------------------------------|----------------------|
| `Loan_default.ipynb`            | EDA notebook         |
| `Loan_default_MLpipeline.ipynb` | ML pipeline notebook |
| `Loan_Default.csv`              | Dataset              |
