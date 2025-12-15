# credit-risk-model_week4
# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability
The Basel II Accord emphasizes "Minimum Capital Requirements," requiring banks to calculate risk-weighted assets. To comply, banks must use models that are rigorous, validated, and interpretable. We cannot use a "black box" model without explanation because we must justify capital reserves to regulators and explain rejection reasons to customers. This prioritizes models like Logistic Regression or interpretable Decision Trees over complex deep learning models.

## 2. Proxy Variable Necessity & Risks
**Why:** We do not have historical loan repayment data (a "Default" label). We only have eCommerce transaction behavior.
**Proxy:** We use RFM (Recency, Frequency, Monetary) analysis to assume that low-activity, low-value customers are "High Risk."
**Risks:** The proxy assumption might be wrong. A customer might buy rarely (Low Frequency) but be very wealthy and reliable. relying on this proxy might lead to "Type I Errors" (rejecting good customers) or "Type II Errors" (approving bad customers who just happen to spend frequently).

## 3. Trade-offs: Interpretable vs. High-Performance Models
| Feature | Logistic Regression (Interpretable) | Gradient Boosting (Complex) |
| :--- | :--- | :--- |
| **Regulation** | High compliance (Basel II friendly) | Harder to validate for regulators |
| **Accuracy** | Generally lower on non-linear data | High accuracy, captures complex patterns |
| **Explanation** | Clear "Scorecard" (WoE points) | Requires SHAP/LIME for explanation |