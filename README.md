# FINOVA AI — Predictive Financial Forecasting

**An AI-powered financial forecasting and risk intelligence system built using machine learning, ensemble regression, and predictive analytics.**

Part of **FINOVA AI — Personal Financial Intelligence**, developed for the FinTechathon 2026 project.

---

## 1. Project Overview

FINOVA AI Predictive Financial Forecasting is an end-to-end machine learning module designed to analyze historical banking transactions and predict future financial behavior.

The system transforms transaction records into structured monthly financial information, learns patterns through optimized machine learning algorithms, and generates next-month financial forecasts.

In addition to forecasting income, expenses, and account balances, the system integrates an independent financial risk classification engine to identify potential negative cash-flow periods.

The module includes trained machine learning models, a FastAPI inference service, and an interactive React dashboard with animated financial visualizations.

**Project Status:** Research prototype developed, trained, evaluated, and integrated into the FINOVA AI application.

## 2. Core Features

| Feature | Description |
|---|---|
| Income Forecasting | Predicts next-month cash inflow |
| Expense Forecasting | Estimates next-month cash outflow |
| Net Cash Flow Prediction | Calculates projected income minus expenses |
| Balance Forecasting | Estimates the next month's account balance |
| Financial Risk Detection | Identifies potential negative cash-flow periods |
| Ensemble Learning | Combines optimized models where validation performance improves |
| Automatic Feature Engineering | Generates historical lag, rolling and seasonal features |
| REST API | Serves predictions through FastAPI |
| Interactive Dashboard | Visualizes financial predictions and historical patterns |
| Model Export | Stores trained models and inference configuration using Joblib |

## 3. Dataset

**Dataset:** Czech Banking Dataset (PKDD'99), using a prepared TSV-based banking dataset.

Three primary files were used:

| Dataset File | Records | Purpose |
|---|---:|---|
| `fin_trans.tsv` | 1,056,320 | Historical financial transactions |
| `fin_account.tsv` | 4,500 | Bank account information |
| `fin_order.tsv` | 6,471 | Standing payment orders |

The dataset version used for this project contains transaction dates ranging from January 2013 to December 2018.

### Data Preprocessing

The preprocessing workflow includes:

- Transaction amount validation and income/expense identification.
- Missing value and duplicate record analysis.
- Account-level transaction history evaluation.
- Monthly financial aggregation.
- Detection and explicit identification of missing calendar months.
- Historical feature engineering and target generation.
- Chronological training, validation, and testing splits.

### Processed Dataset Statistics

| Property | Value |
|---|---:|
| Eligible Accounts | 4,493 |
| Original Monthly Records | 184,903 |
| Completed Monthly Timeline | 185,088 |
| Missing Calendar Months Identified | 185 |
| Records with Complete Forecasting Targets | 180,410 |
| Final Model Input Features | 37 |
| Forecasting Targets | 4 |

Missing historical months are explicitly identified rather than automatically treated as confirmed zero-transaction periods.

## 4. Feature Engineering

The forecasting engine uses 37 model input features derived from monthly financial activity.

### Current Financial Features

- Monthly income
- Monthly expenses
- Net cash flow
- Transaction count
- Last recorded account balance

### Historical Lag Features

Lag features capture financial information from previous months.

For income, expenses, net cash flow, and account balance, the model includes:

`lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`

### Rolling Statistics

Three-month and six-month rolling averages capture recent financial patterns.

Rolling features are generated for:

- Income
- Expenses
- Net cash flow

### Seasonal Features

Calendar-based features capture recurring monthly and quarterly behavior, including cyclical encoding of the target forecast month.

Feature generation is designed to use information available at the forecasting cutoff without using the target month's actual financial values.

## 5. Machine Learning Algorithms

Four regression algorithms were evaluated:

| Algorithm | Application |
|---|---|
| XGBoost Regressor | Nonlinear financial forecasting |
| CatBoost Regressor | Financial patterns and balance estimation |
| LightGBM Regressor | Efficient gradient boosting |
| Elastic Net Regression | Regularized linear benchmark |

### Hyperparameter Optimization

**Optuna** was used to optimize model hyperparameters.

The initial optimization process included:

- 210 hyperparameter trials across three targets.
- Validation-based model selection.
- Early stopping for gradient boosting models.
- Regularization and tree-complexity optimization.
- Chronological data splitting.
- Comparisons against last-month-value forecasting baselines.

Total initial optimization time was approximately **47.87 minutes** on the development machine.

### Selected Forecasting Architecture

After validation and ensemble comparisons, the following configuration was selected:

| Forecasting Target | Selected Model |
|---|---|
| Income | XGBoost |
| Expense | XGBoost + CatBoost ensemble |
| Account Balance | CatBoost + XGBoost ensemble |
| Net Cash Flow | Predicted Income − Predicted Expense |

The expense and balance ensembles use equal model weights.

## 6. Model Evaluation

The prepared financial dataset was separated chronologically.

| Dataset | Period |
|---|---|
| Training | February 2013 – December 2016 |
| Validation | January 2017 – December 2017 |
| Testing | January 2018 – December 2018 |

After final filtering, **126,656 observations** were used for final regression model fitting, with **53,615 observations** included in the 2018 test evaluation.

### Final Forecasting Performance

| Forecast Target | Baseline MAE | Model MAE | RMSE | R² | MAE Improvement |
|---|---:|---:|---:|---:|---:|
| Income | 804.22 | 572.34 | 1,508.08 | 0.452 | 28.83% |
| Expense | 1,058.01 | 658.06 | 1,272.60 | 0.603 | 37.80% |
| Net Cash Flow | 1,136.09 | 627.38 | 1,077.14 | 0.275 | 44.78% |
| Balance | 744.73 | 632.15 | 1,089.92 | 0.765 | 15.12% |

All four forecasting outputs achieved lower MAE than the last-month-value baseline on the evaluated historical test period.

**Key Result:** The selected system reduced Net Cash Flow forecasting MAE by **44.78%** relative to the baseline.

These results describe historical predictive performance, not guaranteed accuracy on modern or previously unseen financial domains.

## 7. Financial Risk Intelligence

A separate CatBoost classification model was developed to identify potential negative cash-flow periods.

### Classification Objective

- Class 0: Zero or positive next-month net cash flow.
- Class 1: Negative next-month net cash flow.

Both XGBoost and CatBoost classifiers were evaluated. CatBoost was selected after validation-based probability-threshold comparison.

**Selected Classification Threshold:** `0.40`

### Risk Classification Performance

| Metric | Final Test Result |
|---|---:|
| Accuracy | 67.59% |
| Precision | 60.36% |
| Recall | 66.14% |
| F1 Score | 63.12% |
| F2 Score | 64.90% |
| ROC-AUC | 0.7466 |
| Average Precision (PR-AUC) | 0.7023 |

### Confusion Matrix

| | Predicted Non-Risk | Predicted Risk |
|---|---:|---:|
| Actual Non-Risk | 21,370 | 9,763 |
| Actual Risk | 7,613 | 14,869 |

The classifier improves risk-event recall compared with the original regression-based warning rule, but still produces false alerts and misses some negative cash-flow periods.

Its output is presented as a **model risk score**, not as a calibrated probability.

## 8. System Architecture

The forecasting module follows a modular machine learning architecture.

```text
Historical Banking Transactions
             |
             v
    Data Preprocessing
             |
             v
 Monthly Financial Aggregation
             |
             v
     Feature Engineering
             |
             v
  Optimized Machine Learning
             |
       +-----+-----+
       |           |
       v           v
 Regression     CatBoost
  Ensemble      Classifier
       |           |
       v           v
  Financial      Risk
  Forecasts      Signals
       |           |
       +-----+-----+
             |
             v
       Joblib Model
             |
             v
       FastAPI Backend
             |
             v
    React Forecasting UI
```

The forecasting and risk classification models are deployed together inside a serialized inference bundle.

## 9. Model Export

The trained models were exported using Joblib.

**Model File:** `finova_forecasting_engine.pkl`

The serialized bundle contains:

- Five trained regression estimators.
- The CatBoost financial risk classifier.
- Selected model configurations and ensemble weights.
- Balance prediction strategy.
- Input feature names and their required order.
- Classification threshold.
- Forecasting metadata.

Previously trained models can be loaded for inference without repeating hyperparameter optimization.

Serialized model files must be loaded only from trusted sources and require compatible versions of their machine learning dependencies.

## 10. FastAPI Backend Integration

The forecasting engine is integrated into the FINOVA AI application using FastAPI.

### Prediction Endpoint

`POST /forecast/predict`

The API receives an account identifier, the last completed month, and monthly financial history.

**Example Request:**

```json
{
  "account_id": "DEMO001",
  "as_of_month": "2017-12",
  "history": [
    {
      "month": "2017-12",
      "income": 2500,
      "expense": 1800,
      "transaction_count": 15,
      "month_end_balance": 5000
    }
  ]
}
```

This one-month example demonstrates the request format only. A longer historical sequence, preferably at least 13 completed months, is recommended for feature availability.

### Prediction Response

A successful API response provides:

- Predicted income
- Predicted expense
- Predicted net cash flow
- Predicted account balance
- Risk model score
- Classification threshold
- Negative cash-flow warning
- Observed historical months
- Model version

The inference service reconstructs the required feature vector and executes the saved machine learning estimators.

### Local Backend Execution

From the application's `backend` directory:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API Documentation:

`http://127.0.0.1:8000/docs`

The existing Financial Dashboard and Subscription Intelligence API routers are preserved within the larger application.

## 11. Premium Forecasting Dashboard

The forecasting module includes an interactive React and TypeScript interface integrated with the FINOVA AI dashboard.

### Dashboard Capabilities

**Financial Forecast Cards**

Displays next-month predictions for income, expenses, net cash flow, and account balance.

**Interactive Financial Trajectory**

Supports three visualization modes:

- Income versus Expense
- Net Cash Flow
- Account Balance

Historical records are visually separated from model-generated forecasts.

**Financial Risk Visualization**

A circular risk gauge displays the classification model score, selected threshold, and warning status.

**Historical Data Management**

Users can manually enter aggregated monthly financial records or import supported CSV files.

**Report Export**

Forecasting results can be exported as JSON reports.

### Interface Design

The frontend follows the FINOVA AI design system:

| Theme | Design |
|---|---|
| Minimal Finance | Clean, light financial interface |
| Liquid Intelligence | Dark, futuristic financial intelligence interface |

UI elements include responsive layouts, animated financial cards, interactive charts, glass-inspired panels, and reduced-motion accessibility support.

## 12. Technology Stack

| Layer | Technologies |
|---|---|
| Programming Language | Python, TypeScript |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, CatBoost, LightGBM, Scikit-learn |
| Hyperparameter Optimization | Optuna |
| Model Serialization | Joblib |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Styling | Tailwind CSS, custom CSS |
| Icons | Lucide React |
| Development Environment | VS Code, Jupyter Notebook |

## 13. Project Workflow

The development process followed these stages:

1. Dataset exploration and transaction integrity validation.
2. Account-level historical readiness evaluation.
3. Monthly financial aggregation and timeline completion.
4. Lag, rolling, and seasonal feature engineering.
5. Next-month prediction target generation.
6. Chronological train-validation-test preparation.
7. Baseline regression benchmarking.
8. Optuna-based hyperparameter optimization.
9. Regression model and ensemble selection.
10. Historical forecasting evaluation.
11. Financial risk classifier
