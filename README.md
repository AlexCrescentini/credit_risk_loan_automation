## Credit Risk Modeling and Loan Automation using Python

Probability of Default (PD) is a central element in credit risk management and a key input for Basel III Pillar 1 capital requirements. This repository shows how to apply ML models to assess the probability of default of clients from the Kaggle dataset "[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)", and how to use them to automate loan decisions within certain risk bands—including the production of regulatory-compliant adverse action reporting.

The project workflow is organized in two steps:

1. **[`credit_default_model.ipynb`](credit_default_model.ipynb)** — Explores and preprocesses the Kaggle "Give Me Some Credit" dataset (150K loan applications). Compares Logistic Regression, KNN, Random Forest, Gradient Boosting, and XGBoost classifiers using AUC-ROC and confusion matrices. Applies Platt scaling to calibrate the best model's predicted probabilities.

2. **[`loan_scoring_and_decision.ipynb`](loan_scoring_and_decision.ipynb)** — Deploys the calibrated XGBoost model (AUC = 0.87) to score synthetic loan applications and automate approve/review/decline decisions via risk bands. Generates SHAP-based adverse action notices for declined applicants in compliance with regulatory requirements.

#### Repository Structure
```
credit_risk_loan_automation/
├── README.md
├── environment.yml
├── data/
├── model_results/
├── credit_default_model.ipynb
└── loan_scoring_and_decision.ipynb
```

#### Packages

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `matplotlib` · `seaborn` · `scipy` · `joblib` · `kagglehub`

---

## Key Results

#### 1. Credit Default Model

The first notebook addresses the full ML pipeline: data cleaning (handling missing values, outliers, and class imbalance), feature scaling with RobustScaler, and model comparison. Five classifiers are trained and evaluated on a stratified 80/20 split. XGBoost achieves the highest validation AUC (0.87) with minimal overfitting. Platt scaling is then applied to calibrate the raw probabilities, reducing Brier score and ensuring that predicted PDs align with observed default rates.

![ROC Curve Comparison](model_results/models_ROC_comparison.png)

![XGBoost Calibration](model_results/XGBoost_calibration.png)

#### 2. Loan Scoring and Decision

The second notebook simulates a production environment. Synthetic loan applications with realistic borrower profiles (prime, near-prime, subprime) are generated and scored using the saved XGBoost model and Platt parameters. Each application is assigned to a risk band and receives an automated decision based on PD thresholds. For declined applicants, SHAP values decompose the prediction into feature contributions, enabling transparent adverse action notices as required by regulation.

![Loan Decision Automated](model_results/loan_decision_automated.png)

![Adverse Action Notice](model_results/adverse_action_notice.png)
