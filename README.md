# Credit Risk Model - Bati Bank

## Credit Scoring Business Understanding

### 1. Basel II & Interpretability
The Basel II Accord requires banks to calculate capital reserves based on risk-weighted assets. To comply, models must be transparent. We cannot use "black box" models (like complex neural networks) because we must explain to regulators and customers why a specific loan was rejected. This makes interpretable models like Logistic Regression (backed by Weight of Evidence) preferable.

### 2. The Proxy Variable (Risk)
**Why:** The Xente dataset lacks a "Loan Default" column. We only have eCommerce transaction history.
**The Proxy:** We define "High Risk" users using RFM (Recency, Frequency, Monetary) analysis.
**Assumption:** Users who are inactive (High Recency) and spend little (Low Frequency/Monetary) are statistically less likely to be reliable borrowers compared to active, high-volume users.
**Risks:** Relying on a proxy introduces "Type I Errors" (rejecting good customers who prefer cash) and "Type II Errors" (approving bad customers who transact frequently but don't repay).

### 3. Model Trade-offs
*   **Logistic Regression:** Highly interpretable, compliant with regulations, easy to deploy, but captures fewer complex patterns.
*   **Gradient Boosting (Random Forest/XGBoost):** Higher accuracy, handles non-linear data well, but harder to interpret for regulatory compliance.