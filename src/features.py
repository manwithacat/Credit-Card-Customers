"""
Feature Engineering Pipeline for Credit Card Customer Churn Analysis

This module takes cleaned data and creates derived features that may be more
predictive of customer churn. Feature engineering is the process of using
domain knowledge to create new features from raw data that make machine learning
algorithms work better.

Pipeline: cleaned.parquet → Feature Engineering → features.parquet
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd  # Data manipulation
import numpy as np   # Numerical operations
from pathlib import Path  # File path handling


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================
def run_feature_engineering():
    """
    Execute feature engineering pipeline to create predictive features.

    This function:
    - Loads cleaned data from ETL output
    - Creates derived features based on domain knowledge
    - Engineers behavioral and risk indicators
    - Saves enriched dataset with new features

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """

    # ========================================================================
    # STEP 1: LOAD CLEANED DATA
    # ========================================================================
    print("📥 Loading cleaned data...")

    # Path to cleaned data (output from etl.py)
    cleaned_path = Path("data/processed/cleaned.parquet")

    # Check if file exists
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}. "
            "Please run src/etl.py first to generate cleaned data."
        )

    df = pd.read_parquet(cleaned_path)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    print("🔧 Engineering features...")

    # ------------------------------------------------------------------------
    # 2.1: RFM METRICS (Recency, Frequency, Monetary)
    # ------------------------------------------------------------------------
    # RFM is a classic customer segmentation technique
    # - Recency: How recently did they interact?
    # - Frequency: How often do they interact?
    # - Monetary: How much do they spend?

    print("  → Creating RFM features...")

    # Frequency: Already have total_trans_ct
    # Monetary: Already have total_trans_amt
    # We'll create normalized versions (0-1 scale) for easier interpretation

    if "total_trans_ct" in df.columns:
        # Normalize transaction count to 0-1 scale
        # min-max scaling: (x - min) / (max - min)
        trans_ct_min = df["total_trans_ct"].min()
        trans_ct_max = df["total_trans_ct"].max()
        df["frequency_score"] = (
            (df["total_trans_ct"] - trans_ct_min) / (trans_ct_max - trans_ct_min)
        ).fillna(0)  # fillna in case all values are the same

    if "total_trans_amt" in df.columns:
        # Normalize transaction amount to 0-1 scale
        trans_amt_min = df["total_trans_amt"].min()
        trans_amt_max = df["total_trans_amt"].max()
        df["monetary_score"] = (
            (df["total_trans_amt"] - trans_amt_min) / (trans_amt_max - trans_amt_min)
        ).fillna(0)

    # Create combined RFM score (average of frequency and monetary)
    if "frequency_score" in df.columns and "monetary_score" in df.columns:
        df["rfm_score"] = (df["frequency_score"] + df["monetary_score"]) / 2

    # ------------------------------------------------------------------------
    # 2.2: ENGAGEMENT SCORE
    # ------------------------------------------------------------------------
    # Combines multiple engagement indicators into a single metric

    print("  → Creating engagement score...")

    # Components of engagement:
    # 1. Number of products (relationship_count)
    # 2. Transaction frequency
    # 3. Months of inactivity (inverse)
    # 4. Contacts count (customer service interaction)

    engagement_components = []

    if "total_relationship_count" in df.columns:
        # More products = higher engagement
        rel_min = df["total_relationship_count"].min()
        rel_max = df["total_relationship_count"].max()
        engagement_rel = (df["total_relationship_count"] - rel_min) / (rel_max - rel_min)
        engagement_components.append(engagement_rel.fillna(0))

    if "months_inactive_12_mon" in df.columns:
        # More inactive months = lower engagement (so we invert it)
        # 1 - normalized_value gives us the inverse
        inactive_min = df["months_inactive_12_mon"].min()
        inactive_max = df["months_inactive_12_mon"].max()
        engagement_inactive = 1 - (
            (df["months_inactive_12_mon"] - inactive_min) / (inactive_max - inactive_min)
        )
        engagement_components.append(engagement_inactive.fillna(0))

    if "frequency_score" in df.columns:
        engagement_components.append(df["frequency_score"])

    # Average all engagement components
    if engagement_components:
        df["engagement_score"] = pd.concat(engagement_components, axis=1).mean(axis=1)

    # ------------------------------------------------------------------------
    # 2.3: CREDIT HEALTH INDICATORS
    # ------------------------------------------------------------------------
    print("  → Creating credit health indicators...")

    # Credit-to-limit ratio (how much of their limit are they using)
    if "total_revolving_bal" in df.columns and "credit_limit" in df.columns:
        df["credit_to_limit_ratio"] = (
            df["total_revolving_bal"] / df["credit_limit"].replace(0, np.nan)
        )

    # Available credit ratio
    if "avg_open_to_buy" in df.columns and "credit_limit" in df.columns:
        df["available_credit_ratio"] = (
            df["avg_open_to_buy"] / df["credit_limit"].replace(0, np.nan)
        )

    # Utilization categories (binned)
    if "utilization" in df.columns:
        # Create categorical version of utilization
        # This is useful for visualizations and some ML algorithms
        df["utilization_category"] = pd.cut(
            df["utilization"],
            bins=[0, 0.3, 0.7, 0.9, float("inf")],
            labels=["Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    # ------------------------------------------------------------------------
    # 2.4: TRANSACTION BEHAVIOR FEATURES
    # ------------------------------------------------------------------------
    print("  → Creating transaction behavior features...")

    # Transaction velocity: spending per transaction
    # Already created in ETL as avg_transaction, but let's add more

    # Transaction frequency per product
    if "total_trans_ct" in df.columns and "total_relationship_count" in df.columns:
        df["trans_per_product"] = (
            df["total_trans_ct"] / df["total_relationship_count"].replace(0, np.nan)
        )

    # Monthly spending rate (total amount / tenure in months)
    if "total_trans_amt" in df.columns and "months_on_book" in df.columns:
        df["monthly_spend_rate"] = (
            df["total_trans_amt"] / df["months_on_book"].replace(0, np.nan)
        )

    # Transaction activity ratio: comparing Q4 to Q1 change
    if "total_ct_chng_q4_q1" in df.columns:
        # Already have the raw change, let's categorize it
        # Positive = increasing activity, Negative = decreasing activity
        df["transaction_trend"] = pd.cut(
            df["total_ct_chng_q4_q1"],
            bins=[-float("inf"), -0.2, 0.2, float("inf")],
            labels=["Decreasing", "Stable", "Increasing"],
        )

    # ------------------------------------------------------------------------
    # 2.5: CUSTOMER LIFECYCLE STAGE
    # ------------------------------------------------------------------------
    print("  → Creating customer lifecycle stage...")

    # Segment customers by tenure
    if "months_on_book" in df.columns:
        df["lifecycle_stage"] = pd.cut(
            df["months_on_book"],
            bins=[0, 12, 24, 36, float("inf")],
            labels=["New (0-1yr)", "Growing (1-2yr)", "Mature (2-3yr)", "Loyal (3yr+)"],
            include_lowest=True,
        )

    # ------------------------------------------------------------------------
    # 2.6: RISK INDICATORS
    # ------------------------------------------------------------------------
    print("  → Creating risk indicators...")

    # Combine multiple risk factors into a risk score
    # Higher score = higher risk of churn

    risk_components = []

    # Low transaction count increases risk
    if "frequency_score" in df.columns:
        risk_components.append(1 - df["frequency_score"])  # Inverse

    # High inactivity increases risk
    if "months_inactive_12_mon" in df.columns:
        inactive_min = df["months_inactive_12_mon"].min()
        inactive_max = df["months_inactive_12_mon"].max()
        inactive_norm = (df["months_inactive_12_mon"] - inactive_min) / (
            inactive_max - inactive_min
        )
        risk_components.append(inactive_norm.fillna(0))

    # Low engagement increases risk
    if "engagement_score" in df.columns:
        risk_components.append(1 - df["engagement_score"])

    # Extreme utilization (very low or very high) increases risk
    if "utilization" in df.columns:
        # U-shaped risk: both very low (<0.1) and very high (>0.9) are risky
        # Create a score where middle values (0.3-0.7) are low risk
        util_risk = df["utilization"].apply(
            lambda x: 1 - abs(x - 0.5) / 0.5 if pd.notna(x) else 0.5
        )
        risk_components.append(util_risk)

    # Calculate average risk score
    if risk_components:
        df["churn_risk_score"] = pd.concat(risk_components, axis=1).mean(axis=1)

    # Create risk categories
    if "churn_risk_score" in df.columns:
        df["risk_category"] = pd.cut(
            df["churn_risk_score"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
            include_lowest=True,
        )

    # ------------------------------------------------------------------------
    # 2.7: CUSTOMER VALUE SEGMENTS
    # ------------------------------------------------------------------------
    print("  → Creating customer value segments...")

    # Combine RFM score and engagement to create value segments
    if "rfm_score" in df.columns and "engagement_score" in df.columns:
        # High RFM + High Engagement = Champions
        # High RFM + Low Engagement = At Risk
        # Low RFM + High Engagement = Potential
        # Low RFM + Low Engagement = Hibernating

        def segment_customer(row):
            """Classify customer into value segment"""
            rfm = row.get("rfm_score", 0.5)
            eng = row.get("engagement_score", 0.5)

            if rfm > 0.6 and eng > 0.6:
                return "Champion"
            elif rfm > 0.6 and eng <= 0.6:
                return "At Risk"
            elif rfm <= 0.6 and eng > 0.6:
                return "Potential"
            else:
                return "Hibernating"

        df["customer_segment"] = df.apply(segment_customer, axis=1)

    # ------------------------------------------------------------------------
    # 2.8: INTERACTION FEATURES (Feature Crosses)
    # ------------------------------------------------------------------------
    print("  → Creating interaction features...")

    # Feature crosses: combining two features to capture interactions
    # These can reveal patterns that single features can't show

    # Credit limit × Utilization = Actual balance used
    if "credit_limit" in df.columns and "utilization" in df.columns:
        df["balance_estimate"] = df["credit_limit"] * df["utilization"]

    # Products × Monthly spend = Spend per product per month
    if (
        "total_relationship_count" in df.columns
        and "monthly_spend_rate" in df.columns
    ):
        df["spend_per_product_monthly"] = (
            df["monthly_spend_rate"]
            / df["total_relationship_count"].replace(0, np.nan)
        )

    # Tenure × Transaction count = Transaction density over lifetime
    if "months_on_book" in df.columns and "total_trans_ct" in df.columns:
        df["transaction_density"] = (
            df["total_trans_ct"] / df["months_on_book"].replace(0, np.nan)
        )

    # ========================================================================
    # STEP 3: DATA VALIDATION
    # ========================================================================
    print("✅ Feature engineering complete!")
    print(f"   Created {df.shape[1] - pd.read_parquet(cleaned_path).shape[1]} new features")

    # Print summary of new features
    new_features = set(df.columns) - set(pd.read_parquet(cleaned_path).columns)
    if new_features:
        print(f"   New features: {', '.join(sorted(new_features))}")

    # ========================================================================
    # STEP 4: SAVE ENRICHED DATA
    # ========================================================================
    features_path = Path("data/processed/features.parquet")
    df.to_parquet(features_path, index=False, engine="pyarrow")
    print(f"💾 Saved feature-enriched data → {features_path}")

    return df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_feature_engineering()
