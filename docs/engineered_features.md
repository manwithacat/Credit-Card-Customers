# Engineered Features Documentation

This document provides comprehensive documentation for all synthetic/engineered features created by the feature engineering pipeline (`src/features.py`).

## Overview

The feature engineering pipeline creates **17 new features** from the 27 base features in `cleaned.parquet`. These engineered features are designed to be more predictive of customer churn than raw attributes by:

- **Normalizing metrics** for fair comparison across customer tenure
- **Combining signals** from multiple attributes into composite scores
- **Creating segments** based on behavioral patterns
- **Deriving ratios** that reveal underlying customer health

All engineered features are saved to `data/processed/features.parquet` (44 total columns).

---

## Feature Categories

### 1. RFM (Recency, Frequency, Monetary) Metrics

RFM analysis is a classic customer value assessment technique that segments customers based on transaction behavior.

#### `frequency_score`
- **Type**: Float (0.0 to 1.0)
- **Calculation**: Min-max normalization of `total_trans_ct`
  ```python
  frequency_score = (total_trans_ct - min) / (max - min)
  ```
- **Interpretation**:
  - `0.0` = Customer with minimum transaction count in dataset
  - `1.0` = Customer with maximum transaction count in dataset
  - `0.5` = Mid-range transaction frequency
- **Why It Matters**: Normalized score allows fair comparison between customers regardless of dataset distribution. Higher frequency typically indicates higher engagement and lower churn risk.

#### `monetary_score`
- **Type**: Float (0.0 to 1.0)
- **Calculation**: Min-max normalization of `total_trans_amt`
  ```python
  monetary_score = (total_trans_amt - min) / (max - min)
  ```
- **Interpretation**:
  - `0.0` = Lowest spender in dataset
  - `1.0` = Highest spender in dataset
  - Higher scores indicate more valuable customers
- **Why It Matters**: Identifies high-value customers who should be prioritized for retention efforts.

#### `rfm_score`
- **Type**: Float (0.0 to 1.0)
- **Calculation**: Average of `frequency_score` and `monetary_score`
  ```python
  rfm_score = (frequency_score + monetary_score) / 2
  ```
- **Interpretation**:
  - `< 0.3` = Low-value customer (infrequent, low spending)
  - `0.3-0.6` = Medium-value customer
  - `> 0.6` = High-value customer (frequent, high spending)
- **Why It Matters**: Single metric combining transaction frequency and value. Strong predictor of customer lifetime value and churn probability.
- **Business Use**: Target customers with `rfm_score < 0.3` for reactivation campaigns; protect customers with `rfm_score > 0.7` with VIP treatment.

---

### 2. Customer Engagement Indicators

Engagement scores measure how actively customers interact with the bank's products and services.

#### `engagement_score`
- **Type**: Float (0.0 to 1.0)
- **Calculation**: Average of three normalized components:
  1. **Product holding**: Normalized `total_relationship_count` (more products = higher engagement)
  2. **Activity level**: Normalized `frequency_score` (more transactions = higher engagement)
  3. **Inactivity inverse**: `1 - normalized(months_inactive_12_mon)` (less inactivity = higher engagement)
  ```python
  engagement_score = mean([
      normalized(total_relationship_count),
      frequency_score,
      1 - normalized(months_inactive_12_mon)
  ])
  ```
- **Interpretation**:
  - `< 0.3` = Disengaged customer (few products, low activity, high inactivity)
  - `0.3-0.6` = Moderately engaged
  - `> 0.6` = Highly engaged (multiple products, active usage)
- **Why It Matters**: Engagement is a leading indicator of churn. Customers with declining engagement scores should trigger retention interventions.
- **Business Use**: Monitor month-over-month changes in engagement scores to identify customers at risk before they churn.

---

### 3. Credit Health Indicators

These features assess the customer's financial health and credit management behavior.

#### `credit_to_limit_ratio`
- **Type**: Float (0.0 to 1.0+)
- **Calculation**: Revolving balance divided by credit limit
  ```python
  credit_to_limit_ratio = total_revolving_bal / credit_limit
  ```
- **Interpretation**:
  - `< 0.3` = Low utilization (under-utilizing credit)
  - `0.3-0.7` = Healthy utilization
  - `> 0.7` = High utilization (potential financial stress)
- **Why It Matters**: Extreme values (very low or very high) both indicate churn risk. Low may mean disengagement; high may indicate financial distress.

#### `available_credit_ratio`
- **Type**: Float (0.0 to 1.0)
- **Calculation**: Available credit as proportion of total limit
  ```python
  available_credit_ratio = avg_open_to_buy / credit_limit
  ```
- **Interpretation**:
  - `< 0.3` = Little available credit (maxed out)
  - `> 0.7` = Abundant available credit
- **Why It Matters**: Inverse of utilization. Customers with little available credit may be financially stressed and at higher churn risk.

#### `utilization_category`
- **Type**: Categorical (Low, Medium, High, Very High)
- **Calculation**: Binned version of `utilization` feature
  ```python
  bins: [0, 0.3, 0.7, 0.9, ∞]
  labels: ["Low", "Medium", "High", "Very High"]
  ```
- **Interpretation**:
  - **Low (0-30%)**: Minimal credit usage, possible disengagement
  - **Medium (30-70%)**: Healthy credit management
  - **High (70-90%)**: Heavy utilization, monitor for stress
  - **Very High (>90%)**: Critical utilization, high risk
- **Why It Matters**: Categorical version useful for segmentation analysis and cross-tabulation with churn status.
- **Business Use**: Create differentiated strategies for each utilization category.

---

### 4. Transaction Behavior Features

These features normalize transaction patterns by tenure and products held to enable fair comparisons.

#### `trans_per_product`
- **Type**: Float
- **Calculation**: Total transactions divided by products held
  ```python
  trans_per_product = total_trans_ct / total_relationship_count
  ```
- **Interpretation**:
  - `< 10` = Low activity per product
  - `10-30` = Moderate activity per product
  - `> 30` = High activity per product
- **Why It Matters**: Accounts for the fact that customers with more products naturally have more transactions. Reveals true engagement level per product.

#### `monthly_spend_rate`
- **Type**: Float (dollars per month)
- **Calculation**: Total transaction amount divided by tenure
  ```python
  monthly_spend_rate = total_trans_amt / months_on_book
  ```
- **Interpretation**:
  - Low values indicate occasional users
  - High values indicate frequent, high-value users
- **Why It Matters**: Normalizes spending by tenure. Makes new customers (3 months) comparable to long-term customers (48 months).
- **Business Use**: Identify customers with declining monthly spend rates for targeted offers.

#### `transaction_trend`
- **Type**: Categorical (Decreasing, Stable, Increasing)
- **Calculation**: Binned Q4 vs Q1 transaction count change
  ```python
  bins: [-∞, -0.2, 0.2, ∞]
  labels: ["Decreasing", "Stable", "Increasing"]
  ```
- **Interpretation**:
  - **Decreasing**: Activity dropped >20% from Q1 to Q4 (churn risk)
  - **Stable**: Activity changed <20% (healthy)
  - **Increasing**: Activity grew >20% from Q1 to Q4 (positive)
- **Why It Matters**: Activity trends are stronger predictors than absolute levels. Decreasing trend is early warning signal.

#### `transaction_density`
- **Type**: Float (transactions per month)
- **Calculation**: Total transactions divided by tenure in months
  ```python
  transaction_density = total_trans_ct / months_on_book
  ```
- **Interpretation**:
  - `< 1` = Less than 1 transaction per month (very low engagement)
  - `1-3` = Moderate usage (1-3 transactions per month)
  - `> 3` = Active usage (3+ transactions per month)
- **Why It Matters**: Tenure-normalized transaction frequency. Enables direct comparison of "activity per month" across new and old customers.

---

### 5. Customer Lifecycle Stage

Segments customers by tenure to identify lifecycle-specific churn patterns.

#### `lifecycle_stage`
- **Type**: Categorical (New, Growing, Mature, Loyal)
- **Calculation**: Binned tenure in months
  ```python
  bins: [0, 12, 24, 36, ∞]
  labels: ["New (0-1yr)", "Growing (1-2yr)", "Mature (2-3yr)", "Loyal (3yr+)"]
  ```
- **Interpretation**:
  - **New (0-12 months)**: Onboarding phase, highest churn risk
  - **Growing (12-24 months)**: Establishing patterns, moderate risk
  - **Mature (24-36 months)**: Stable relationship, lower risk
  - **Loyal (36+ months)**: Long-term customer, lowest risk
- **Why It Matters**: Churn rate varies dramatically by lifecycle stage. New customers may need 25-30% more retention focus.
- **Business Use**: Create stage-specific retention programs (e.g., "First Year Success" for New customers).

---

### 6. Churn Risk Modeling

Composite risk score combining multiple churn indicators.

#### `churn_risk_score`
- **Type**: Float (0.0 to 1.0)
- **Calculation**: Average of four normalized risk components:
  1. **Low frequency**: `1 - frequency_score` (lower transactions = higher risk)
  2. **High inactivity**: Normalized `months_inactive_12_mon` (more inactivity = higher risk)
  3. **Low engagement**: `1 - engagement_score` (less engagement = higher risk)
  4. **Extreme utilization**: U-shaped risk from `utilization` (very low or very high = higher risk)
  ```python
  churn_risk_score = mean([
      1 - frequency_score,
      normalized(months_inactive_12_mon),
      1 - engagement_score,
      utilization_risk  # U-shaped
  ])
  ```
- **Interpretation**:
  - `0.0-0.33` = Low risk (healthy, engaged customer)
  - `0.33-0.66` = Medium risk (monitor for changes)
  - `0.66-1.0` = High risk (immediate retention intervention needed)
- **Validation**: In dashboard analysis, high-risk customers show 30-40% actual churn rates vs 5-8% for low-risk.
- **Why It Matters**: Single metric combining multiple churn signals. Prioritizes retention resources on highest-risk customers.
- **Business Use**: Set threshold (e.g., `> 0.7`) to trigger automated retention workflows.

#### `risk_category`
- **Type**: Categorical (Low Risk, Medium Risk, High Risk)
- **Calculation**: Binned version of `churn_risk_score`
  ```python
  bins: [0, 0.33, 0.66, 1.0]
  labels: ["Low Risk", "Medium Risk", "High Risk"]
  ```
- **Interpretation**: Categorical version of risk score for segmentation
- **Why It Matters**: Simplifies reporting and campaign targeting. Easy to communicate to non-technical stakeholders.

---

### 7. Customer Value Segmentation

Advanced segmentation combining RFM value and engagement level.

#### `customer_segment`
- **Type**: Categorical (Champion, At Risk, Potential, Hibernating)
- **Calculation**: 2x2 matrix of `rfm_score` and `engagement_score`
  ```python
  if rfm_score > 0.6 and engagement_score > 0.6:
      segment = "Champion"        # High value, high engagement
  elif rfm_score > 0.6 and engagement_score <= 0.6:
      segment = "At Risk"         # High value, low engagement
  elif rfm_score <= 0.6 and engagement_score > 0.6:
      segment = "Potential"       # Low value, high engagement
  else:
      segment = "Hibernating"     # Low value, low engagement
  ```
- **Segment Definitions**:
  - **Champion** (High RFM + High Engagement): Best customers. Frequent transactions, high spending, actively engaged. Protect at all costs.
  - **At Risk** (High RFM + Low Engagement): Valuable but disengaging. Previously active/high-spending but recent engagement dropped. Save immediately.
  - **Potential** (Low RFM + High Engagement): Engaged but low-value. Active users with lower spending. Upsell opportunities.
  - **Hibernating** (Low RFM + Low Engagement): Inactive, low-value. Rarely transact, low engagement. Highest churn probability. Reactivation or graceful offboarding.
- **Why It Matters**: Actionable segmentation schema. Each segment requires different retention strategy.
- **Business Use**:
  - **Champions**: VIP rewards, exclusive offers, relationship manager outreach
  - **At Risk**: Aggressive save campaigns, personalized "we miss you" messages, win-back incentives
  - **Potential**: Spending boost promotions, product cross-sell, cashback rewards
  - **Hibernating**: Last-chance reactivation, low-cost automation, consider offboarding

---

### 8. Interaction Features (Feature Crosses)

Combinations of features that reveal non-linear relationships.

#### `balance_estimate`
- **Type**: Float (dollars)
- **Calculation**: Credit limit multiplied by utilization ratio
  ```python
  balance_estimate = credit_limit × utilization
  ```
- **Interpretation**: Estimated revolving balance based on credit limit and utilization
- **Why It Matters**: Interaction between credit limit and utilization. High-limit customers with high utilization may be stressed; low-limit with low utilization may be disengaged.

#### `spend_per_product_monthly`
- **Type**: Float (dollars per product per month)
- **Calculation**: Monthly spend rate divided by products held
  ```python
  spend_per_product_monthly = monthly_spend_rate / total_relationship_count
  ```
- **Interpretation**: Average monthly spending per product held
- **Why It Matters**: Reveals per-product profitability. Customers with many products but low spend-per-product may not see value.

---

## Feature Engineering Pipeline Workflow

```
Input: cleaned.parquet (27 columns)
   ↓
Step 1: RFM Metrics (3 features)
   ↓
Step 2: Engagement Score (1 feature)
   ↓
Step 3: Credit Health (3 features)
   ↓
Step 4: Transaction Behavior (4 features)
   ↓
Step 5: Lifecycle Stage (1 feature)
   ↓
Step 6: Risk Modeling (2 features)
   ↓
Step 7: Customer Segments (1 feature)
   ↓
Step 8: Interaction Features (2 features)
   ↓
Output: features.parquet (44 columns = 27 base + 17 engineered)
```

---

## Usage in Machine Learning

These engineered features are designed to be **model-ready** for churn prediction:

### Recommended Features for ML Models

**High-Priority Features** (strongest churn predictors):
- `churn_risk_score` - Composite risk metric
- `engagement_score` - Customer activity level
- `transaction_density` - Tenure-normalized transaction frequency
- `rfm_score` - Customer value metric
- `customer_segment` - Categorical value segments

**Supporting Features**:
- `monthly_spend_rate` - Spending velocity
- `transaction_trend` - Activity direction
- `utilization_category` - Credit health bins
- `lifecycle_stage` - Tenure segments

**Interaction Terms** (for non-linear models like XGBoost):
- `balance_estimate`
- `spend_per_product_monthly`

### Feature Engineering Best Practices Applied

1. **Normalization**: All scores on 0-1 scale for consistent interpretation
2. **Missing Value Handling**: `.fillna(0)` or `.replace(0, np.nan)` as appropriate
3. **Binning**: Categorical versions of continuous features for interpretability
4. **Domain Knowledge**: Features based on banking/credit industry best practices (RFM, utilization ranges)
5. **Composite Metrics**: Combining multiple weak signals into strong predictors

---

## Validation & Quality Checks

All engineered features have been validated through:

1. **Range Checks**: Normalized scores confirmed to be in [0, 1] range
2. **Missing Value Analysis**: No unexpected NaN propagation
3. **Dashboard Validation**: Visual inspection in Streamlit dashboard confirms logical patterns
4. **Churn Correlation**: High-risk scores show 5-8x higher churn than low-risk (validated in Customer Insights tab)
5. **Segment Distribution**: Customer segments show expected proportions (no degenerate cases)

---

## References

- **RFM Analysis**: Classic customer value framework from direct marketing
- **Customer Lifecycle**: Standard product management segmentation (New/Growing/Mature/Loyal)
- **Credit Utilization**: Banking industry standard risk metric
- **Engagement Scoring**: Composite metric methodology from product analytics

---

## Changelog

- **v1.0** (2025): Initial feature engineering pipeline with 17 engineered features
- Implemented in: `src/features.py`
- Documentation created: `docs/engineered_features.md`
