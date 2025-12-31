## Customer-Default-Risk-Modeling---Finance-Style-End-to-End-Project
This repository demonstrates a production-style *Probability of Default (PD)* modeling workflow using transactional data and *proxy default labels* (when true repayment/default labels are unavailable). The project is structured to reflect how a bank/fintech would build, validate, interpret, and monitor a risk model.
*Note:* This is an educational/portfolio project. The “default” label is a *proxy* derived from observed loss-related behaviors, and the model is intended for *risk ranking / early-warning*, not automated lending decisions.
PD (Probability of Default) is the probability that a customer will fail to meet a financial obligation within a defined time horizon.
In many real-world situations:
The product is new
Historical repayment data doesn’t exist
Credit bureau data is unavailable
The model is needed before defaults are fully observed. So Waiting for true defaults can take months or years.
In fact Proxy PD modeling estimates default risk using observable behaviors that are strongly correlated with future loss, even if actual default outcomes are not yet available.
Instead of modeling default directly, we model default-like behavior.
what makes this project a proxy PD model:
True repayment default is not available.
so, we defined default using:
Extreme return behavior, Negative profitability, Aggressive discount usage
This is a loss-based behavioral proxy, which is common in trade credit, used in early risk screening and suitable for ranking and monitoring.
Key risks in proxy PD modeling:
Risk 1: Target leakage
I removed features used in target definition.
Risk 2: Unrealistic bad rates
I calibrated the proxy to ~5.3%
Risk 3: Overconfidence in metrics
I accepted moderate AUC (~0.68)
Risk 4: Misuse of model
I clearly scoped the model as advisory.
Proxy PD modeling is used when true default outcomes are unavailable. Instead of waiting for realized defaults, we define 
a proxy target based on observable loss-related behaviors that are predictive of future default.
The model is positioned as an early-warning and risk-ranking tool, not a regulatory PD.
## 1) Problem
Business problem:
The organization lacks a reliable way to identify customers at elevated risk of financial loss using available transactional data.
Constraints:
1. No true repayment/default labels
2. No credit bureau data
3. Need for explainability and governance
4. Risk of data leakage
5. Model must be monitorable over time
My Solution (as a Data Scientist)
I designed an end-to-end, governance-ready risk scoring system that estimates Probability of Default (PD) using proxy labels, produces interpretable risk rankings, and includes monitoring controls to ensure long-term reliability.
The solution consists of 
1. A defensible proxy for default risk
Because true default data was unavailable, we
Defined proxy default behavior based on observed loss signals
Calibrated the proxy to a realistic 5.33% portfolio bad rate
Documented assumptions and limitations clearly
@ This converts raw data into a usable risk signal.
2. A model-ready customer risk dataset
I transformed raw transactional data into:
- Customer-level behavioral features
- Stability indicators (tenure, frequency)
- Leakage-controlled variables
@ This creates a clean foundation for risk modeling.
3. A transparent baseline risk model (Champion)
- I built a regularized Logistic Regression model that:
- Produces interpretable coefficients
- Ranks customers by risk
- Avoids target leakage
- Achieves realistic discrimination (AUC ≈ 0.68)
@ This model is explainable, stable, and governance-friendly.
4. A performance challenger with explainability
- I evaluated XGBoost to capture non-linear risk patterns and:
- Compared performance honestly against logistic regression
- Used SHAP for global and individual explanations
- Positioned XGBoost as a challenger, not a replacement
@ This balances performance and transparency.
5. Monitoring and risk controls (production thinking)
I implemented:
- Population Stability Index (PSI) for feature and score drift
- Defined monitoring thresholds and retraining triggers
- Separated feature drift from score drift
@ This ensures the solution remains valid over time.
Final Solution:
As a data scientist, my solution is to design a risk-aware, interpretable probability-of-default modeling pipeline that transforms transactional data into actionable risk rankings, while explicitly managing data limitations, leakage, explainability, and model drift. The solution is intended for early-warning and decision support, not automated credit approval.
#################################### 
Problem Statement: 
Build a model to estimate *PD (Probability of Default)* at the *customer level*, enabling:
- Risk ranking (who is riskiest)
- Early-warning monitoring
- Investigation triggers and policy adjustments

  Data and Modeling 
- Data source: customer transaction history (retail-like dataset)
- Modeling unit: *Customer-level aggregated table*
- Output: model-ready dataset with engineered behavioral and stability features

Step-by-Step Process:
 Step 1 — Create Customer-Level Modeling Table (Feature Engineering)
I transformed raw order-level records into a customer risk table, including:
- total_orders, total_sales
- avg_shipping_delay
- customer_tenure_days
- orders_per_year (stability proxy)
- (behavioral proxy features used for target construction)
Output dataset: customer_pd_dataset.csv

Step 2 — Define Proxy Default (Target Engineering)
True repayment/default outcomes were not available, so I defined a *proxy default* using observed loss-related behaviors.

Initial proxy definitions produced:
 *Too sparse* bad rate (~1.54%), difficult for stable modeling
 *Too broad* bad rate (~22%), unrealistic for PD modeling

 Final calibrated proxy target (default_flag_v3) achieved:
- *Bad rate:* ~*5.33%* (realistic for trade-credit / fintech early-warning contexts)

*Final target rule (default_flag_v3):*
- Return rate ≥ 40%  
*OR*
- Return rate ≥ 20% AND total profit < 0 AND avg discount ≥ 30%

This calibration step mirrors how risk teams balance *business realism* and *model learnability*.

# Risk Insight: Order Volume vs Default Risk
<img width="876" height="682" alt="image" src="https://github.com/user-attachments/assets/320b5e7f-4894-46c1-8f5f-4e01e39dd2cd" />
The chart shows default rate by quantile bins of total customer orders. Default risk increases from low to moderate order volumes and then declines for the highest order volumes, indicating a non-monotonic relationship. Customers with moderate activity exhibit the highest risk, likely reflecting sufficient exposure to generate losses without the stabilizing effects observed in highly active, long-tenured customers. This pattern is consistent with behavioral credit risk dynamics and motivates the use of both linear and non-linear models.

What the chart shows:

- X-axis: Quantile bins of total_orders (from low → high)
- Y-axis: Bad rate (default rate)
- Each point represents customers grouped by order volume

Observed pattern:
- Customers with very low order counts have the lowest bad rate
- Bad rate increases for customers with moderate order volume
- Bad rate declines slightly for customers with the highest order volume
This creates a non-monotonic (inverted-U) relationship.

What this means (risk interpretation):
This pattern suggests two different customer behaviors:
1. Low-order customers
Limited exposure
Fewer opportunities to generate loss
Appear low risk simply due to low activity
2.  Mid-order customers (highest risk)
Enough activity to generate returns, discounts, and operational issues
Not yet stable or established
Represent the highest behavioral risk segment
3. High-order customers
High engagement and repeat behavior
More stable relationship
Lower bad rate despite higher exposure
@ This is a classic stability vs exposure trade-off seen in commercial and behavioral risk modeling.

Why this is important (modeling decision):
The relationship is not strictly monotonic
This explains why:
Logistic regression captures the trend only partially
XGBoost can model this pattern more flexibly
It also justifies keeping total_orders as a supporting feature, not a primary risk driver.


Step 3 — Finance-Style EDA (Model Readiness Checks)
I performed EDA focused on risk-model requirements:
- Bad rate sanity checks
- Feature distributions & outliers
- *Bad rate by quantile bins* (monotonicity and stability)
- Leakage risk assessment

Notebook: step2and3.ipynb

Step 4 — Baseline PD Model (Logistic Regression — Champion)
I trained a regularized Logistic Regression baseline model as the *transparent champion*:
- Standardization + median imputation
- Class imbalance handling (`class_weight="balanced"`)
- Evaluation with ROC-AUC, PR-AUC, KS, and calibration

  ![98vb](https://github.com/user-attachments/assets/13b067df-6d79-47e3-852f-6dddf1df9be4)


 Leakage check: Initial near-perfect scores indicated *target imprinting* (proxy label constructed from same features).
 Fix: excluded target-defining variables (e.g., return_rate, avg_discount, total_profit) from model features.

Logistic results (leakage-safe):
- ROC-AUC: **0.6755**
- PR-AUC: **0.3352**

Notebook: step4.ipynb

total_orders (coef = 0.1811)
Higher order volume increases default risk moderately. It reflects increased exposure and operational complexity. Consistent with EDA showing mid-volume customers are riskiest. 

customer_tenure_days (coef = 0.1405)
Shorter tenure -> higher risk
Longer relationships reduce default probability. This is a classic stability variable in credit risk.

total_sales (coef = 0.1216)
Higher sales volume slightly increases risk.
Likely captures exposure rather than instability and effect is smaller than total_orders, which is appropriate.

orders_per_year (coef = 0.1013)
Higher order frequency increases risk marginally. 
Complements total_orders
Captures activity intensity

avg_shipping_delay (coef = 0.0089)
Very small coefficient
Weak standalone predictor
Likely contributes through interactions (better captured by XGBoost)

The regularized logistic regression highlights exposure and stability as the primary drivers of default risk. Customers with higher order volume and shorter tenure exhibit elevated risk, while operational factors such as shipping delays have comparatively weaker effects. Regularization ensures coefficient stability and mitigates overfitting.


 Step 5 — Challenger Model (XGBoost + SHAP)
I trained an *XGBoost challenger* to test whether non-linear relationships improve ranking.
To ensure explainability, I used *SHAP* for:
- Global feature importance (mean |SHAP|)
- Local (individual) explanations

XGBoost results:
- ROC-AUC: **0.6729**
- PR-AUC: **0.3418**

Outcome: XGBoost provided *slight PR-AUC lift*, but similar ROC-AUC to logistic.  
Given governance requirements, Logistic remains champion; XGBoost is retained as a challenger with SHAP explanations.

Notebook: step5.ipynb

Step 6 — Drift Monitoring (PSI)
I implemented Population Stability Index (PSI) for:
- Feature drift monitoring
- Score drift monitoring (predicted PD distribution)

PSI thresholds used:
- < 0.10 = Stable
- 0.10–0.25 = Monitor
- > 0.25 = Investigate / retrain

Notebook: step6.ipynb

 <img width="821" height="652" alt="image" src="https://github.com/user-attachments/assets/43a06a71-e4bc-4b5f-aab0-51f93294f077" />
 <img width="859" height="652" alt="image" src="https://github.com/user-attachments/assets/75fd2097-2ea5-426c-8b0b-0cbdf8e84500" />
 <img width="848" height="652" alt="image" src="https://github.com/user-attachments/assets/4d07465d-d6ea-479b-9113-0b91402b2de8" />


 Governance Documentation
A regulatory-style Model Card is included:
- Model purpose & scope
- Proxy target definition
- Leakage controls
- Performance metrics
- Explainability (SHAP)
- Monitoring & retraining triggers
- Limitations and ethical considerations

See: model_card.md

Key Takeaways
- Proxy PD modeling requires careful target design and calibration
- Leakage detection is critical (perfect metrics are a red flag)
- Logistic regression provides governance-friendly transparency
- XGBoost can be used as a challenger when explainability (SHAP) is included
- Drift monitoring (PSI) is essential for production readiness.

How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt








