# Credit Risk Model - Bati Bank

## Project Overview

This project implements an end-to-end Machine Learning system for Credit Risk Scoring at Bati Bank. The core innovation is the use of alternative transactional data (Recency, Frequency, Monetary - RFM) to engineer a proxy for credit risk, enabling the bank to launch a Buy-Now-Pay-Later (BNPL) service. The solution adheres to MLOps best practices, including modular code, MLflow tracking, unit testing, and a containerized FastAPI for deployment via CI/CD.

## 1. Project Structure (Mandated)

The repository follows a standardized, modular structure:

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD pipeline (Flake8, pytest)
├── data/                       
│   ├── raw/                   # Raw data.csv (added to .gitignore)
│   └── processed/             # Processed train_data.csv
├── notebooks/
│   └── eda.ipynb          # Exploratory Data Analysis & Insights
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature Engineering & Proxy Target (RFM/K-Means)
│   ├── train.py               # Model Training, Hyperparameter Tuning, MLflow Tracking
│   ├── predict.py             # (Optional: for batch inference if needed)
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Data models for API request/response
├── tests/
│   └── test_data_processing.py # Unit tests for data logic
├── Dockerfile                  # Container definition for the API
├── docker-compose.yml          # Simplifies building and running Docker service
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md                   # This file
```

## 2. Local Setup and Execution

### A. Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adem-mekonnen/credit-risk-model_week4.git
    cd credit-risk-model_week4
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # (Linux/Mac)
    # .venv\Scripts\activate   # (Windows CMD)
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Placement:**
    *   Place your raw data file (`data.csv`) into the `data/raw/` directory.

### B. Project Execution Order

The project must be run sequentially:

1.  **Data Processing & Feature Engineering (Task 3 & 4):**
    ```bash
    python src/data_processing.py
    ```
    *Output:* Creates `data/processed/train_data.csv` and the `is_high_risk` target.

2.  **Model Training & MLflow Tracking (Task 5):**
    ```bash
    python src/train.py
    ```
    *Output:* Logs experiments to `./mlruns` and saves the best model artifact to `src/api/best_model.pkl`.

3.  **Run Unit Tests (Task 5 & 6):**
    ```bash
    pytest
    ```

4.  **Run API Locally (Task 6):**
    ```bash
    uvicorn src.api.main:app --reload
    ```
    *Access API documentation at:* `http://127.0.0.1:8000/docs`

## 3. Credit Scoring Business Understanding (Task 1)

### 1. Basel II and Model Interpretability
The Basel II Capital Accord mandates that financial institutions must calculate capital requirements based on their risk exposures. To comply, models used for core lending decisions cannot be opaque "black boxes."
*   **Influence:** This forces the use of **interpretable and well-documented models** (like Logistic Regression with WoE) that allow for clear audit trails, validation, and justification of risk scores. Bati Bank must be able to explain to regulators *and* customers precisely why a loan was approved or declined.

### 2. Proxy Variable Necessity & Risks
*   **Necessity:** We lack a direct "loan default" label. Therefore, a **proxy target variable** is created by transforming transactional behavior (Recency, Frequency, Monetary - RFM) into a risk signal.
*   **Risks:** Making predictions based on a proxy introduces substantial business risk:
    *   **Type I Error (False Positive/Rejecting Good Customers):** A wealthy customer who transacts rarely (low RFM score) is labeled High Risk. Rejecting them means Bati Bank loses profitable business.
    *   **Type II Error (False Negative/Approving Bad Customers):** A customer who makes many small, low-risk purchases (high RFM score) is labeled Low Risk, but defaults on the BNPL loan. This leads to direct financial loss.

### 3. Trade-offs: Interpretable vs. High-Performance Models

| Feature | Logistic Regression (WoE - Interpretable) | Gradient Boosting (Random Forest - Complex) |
| :--- | :--- | :--- |
| **Regulation** | High compliance, ideal for Basel II. | Difficult to validate and explain to regulators. |
| **Accuracy** | Generally lower, best for linear relationships. | High accuracy, captures complex, non-linear interactions. |
| **Explanation** | Clear Scorecard, direct impact of features (WoE). | Requires post-hoc techniques (SHAP/LIME) for justification. |

## 4. Technical Implementation Summary (Tasks 2-6)

| Task | Objective | Implementation Detail |
| :--- | :--- | :--- |
| **Task 2 (EDA)** | Explore data and identify insights. | Performed in `notebooks/eda.ipynb`. Key findings included heavy data skewness (Amount/Value), multicollinearity (Amount/Value), and categorical dominance (e.g., ChannelId\_3). |
| **Task 3 & 4 (FE/Target)** | Create features and the proxy target. | **RFM Clustering** (K-Means) used on scaled Recency, Frequency, and Monetary metrics to define the `is_high_risk` target. Customer-level aggregation (Total/Avg Amount, Count) and Label Encoding were applied. |
| **Task 5 (Training)** | Develop and track models. | **Logistic Regression** (baseline) and **Random Forest** (challenger) were trained. **MLflow** was used to log model metrics (AUC-ROC, Accuracy) and artifacts, with the best model being saved locally for the API. |
| **Task 6 (MLOps)** | Deployment and CI/CD. | A **FastAPI** service (`src/api/main.py`) loads the best model. The service is containerized via **Dockerfile** and runs via `docker-compose`. **GitHub Actions (`ci.yml`)** is configured to run **Flake8** (linter) and **Pytest** (unit tests) on every push, ensuring code quality and reliability. |
