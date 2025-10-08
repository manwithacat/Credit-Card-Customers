"""
ETL (Extract, Transform, Load) Pipeline for Credit Card Customer Data

This module handles the complete data processing pipeline:
1. EXTRACT: Load raw CSV data from data/raw/
2. TRANSFORM: Clean, standardize, and enrich the data
3. LOAD: Save processed data as parquet to data/processed/

The pipeline is designed to be simple and pedagogical, avoiding overengineering
while maintaining good practices for data cleaning and transformation.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd  # Main library for data manipulation and analysis
from pathlib import Path  # Modern, cross-platform way to handle file paths


# ============================================================================
# MAIN ETL FUNCTION
# ============================================================================
def run_etl():
    """
    Execute the complete ETL pipeline to process credit card customer data.

    This function:
    - Loads the raw CSV data
    - Cleans and standardizes column names
    - Enforces proper data types (integers, floats, strings)
    - Handles missing values appropriately
    - Creates derived features (churn_flag, avg_transaction, etc.)
    - Removes duplicate records
    - Saves the cleaned data as a parquet file

    Returns:
        pd.DataFrame: The cleaned and processed dataframe
    """

    # ========================================================================
    # STEP 1: SETUP - Define paths and create directories
    # ========================================================================
    # Path to the raw data file (source CSV provided in the repo)
    raw_path = Path("data/raw/BankChurners.csv")

    # Directory where we'll save the processed output
    processed_dir = Path("data/processed")

    # Create the output directory if it doesn't exist
    # parents=True: create intermediate directories if needed
    # exist_ok=True: don't raise error if directory already exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 2: EXTRACT - Load raw data from CSV
    # ========================================================================
    print("📥 Loading raw data...")
    df = pd.read_csv(raw_path)
    print(f"✅ Raw data loaded: {df.shape}")  # Show (rows, columns)

    # ========================================================================
    # STEP 3: TRANSFORM - Clean and process the data
    # ========================================================================
    print("🧹 Cleaning data...")

    # ------------------------------------------------------------------------
    # 3.1: Remove unnamed/junk columns
    # ------------------------------------------------------------------------
    # Sometimes CSV exports include extra columns named "Unnamed: 0", etc.
    # The ~ operator means "NOT" - so this keeps columns that DON'T match the pattern
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # ------------------------------------------------------------------------
    # 3.2: Standardize column names
    # ------------------------------------------------------------------------
    # Convert column names to lowercase with underscores for consistency
    # This makes them easier to work with and follows Python naming conventions
    # Example: "Customer Age" → "customer_age"
    df.columns = (
        df.columns.str.strip()           # Remove leading/trailing whitespace
        .str.replace(" ", "_")            # Spaces become underscores
        .str.replace("-", "_")            # Hyphens become underscores
        .str.replace("/", "_")            # Slashes become underscores
        .str.lower()                      # Convert to lowercase
    )

    # ------------------------------------------------------------------------
    # 3.3: Clean string/text columns
    # ------------------------------------------------------------------------
    # For all text columns (dtype='object'), strip whitespace from values
    # This prevents issues like "Male" vs "Male " being treated as different
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # ------------------------------------------------------------------------
    # 3.4: Create binary churn flag for modeling
    # ------------------------------------------------------------------------
    # The original "Attrition_Flag" column has text values like "Attrited Customer"
    # We create a numeric binary flag: 1 = churned, 0 = active
    # This is more useful for machine learning models
    if "attrition_flag" in df.columns:
        df["churn_flag"] = df["attrition_flag"].map({
            "Attrited Customer": 1,  # Customer left (churned)
            "Existing Customer": 0   # Customer still active
        }).astype("Int64")  # Use nullable integer type (capital I)

    # ------------------------------------------------------------------------
    # 3.5: Handle missing values in categorical columns
    # ------------------------------------------------------------------------
    # Replace "Unknown" text with pandas NA (proper missing value marker)
    # This allows us to handle missing data more systematically
    for col in ["education_level", "income_category", "marital_status"]:
        if col in df.columns:
            df[col] = df[col].replace({"Unknown": pd.NA, "unknown": pd.NA})

    # ------------------------------------------------------------------------
    # 3.6: Enforce integer data types
    # ------------------------------------------------------------------------
    # These columns should be whole numbers (counts, months, etc.)
    # pd.to_numeric with errors='coerce' converts invalid values to NaN
    # Int64 is nullable integer type (allows NaN, unlike int64)
    int_like = [
        "dependent_count",           # Number of dependents
        "months_on_book",            # Tenure with bank
        "months_inactive_12_mon",    # Inactive months in last year
        "contacts_count_12_mon",     # Number of contacts in last year
        "total_relationship_count",  # Number of products held
        "total_trans_ct"             # Total transaction count
    ]
    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------------
    # 3.7: Enforce float data types
    # ------------------------------------------------------------------------
    # These columns should be decimal numbers (amounts, ratios, etc.)
    float_like = [
        "total_trans_amt",        # Total transaction amount
        "credit_limit",           # Credit card limit
        "total_revolving_bal",    # Revolving balance
        "avg_open_to_buy",        # Average available credit
        "total_amt_chng_q4_q1",   # Amount change Q4 to Q1
        "total_ct_chng_q4_q1",    # Count change Q4 to Q1
        "avg_utilization_ratio"   # Credit utilization ratio
    ]
    for c in float_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ------------------------------------------------------------------------
    # 3.8: Create derived feature - Average Transaction Size
    # ------------------------------------------------------------------------
    # This new feature gives us insight into spending patterns
    # avg_transaction = total_amount / total_count
    # .replace(0, pd.NA) prevents division by zero errors
    if {"total_trans_amt", "total_trans_ct"}.issubset(df.columns):
        df["avg_transaction"] = (
            df["total_trans_amt"] / df["total_trans_ct"].replace(0, pd.NA)
        )

    # ------------------------------------------------------------------------
    # 3.9: Create derived feature - Tenure in Years
    # ------------------------------------------------------------------------
    # Convert months to years for more intuitive interpretation
    # Round to 2 decimal places for readability
    if "months_on_book" in df.columns:
        df["tenure_years"] = (df["months_on_book"].astype("float") / 12).round(2)

    # ------------------------------------------------------------------------
    # 3.10: Standardize utilization ratio
    # ------------------------------------------------------------------------
    # Create a consistent "utilization" column
    # If avg_utilization_ratio exists, use it directly
    # Otherwise, calculate it from revolving_bal / (revolving_bal + open_to_buy)
    if "avg_utilization_ratio" in df.columns:
        df["utilization"] = df["avg_utilization_ratio"]
    elif {"avg_open_to_buy", "total_revolving_bal"}.issubset(df.columns):
        denom = df["avg_open_to_buy"] + df["total_revolving_bal"]
        df["utilization"] = df["total_revolving_bal"] / denom.replace(0, pd.NA)

    # ------------------------------------------------------------------------
    # 3.11: Remove duplicate records
    # ------------------------------------------------------------------------
    # Keep track of how many duplicates we remove for reporting
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"🧽 Duplicates removed: {removed}")

    # ========================================================================
    # STEP 4: LOAD - Save processed data as parquet
    # ========================================================================
    # Parquet format is:
    # - Faster to read than CSV
    # - Smaller file size (compressed)
    # - Preserves data types (CSV doesn't)
    # - Industry standard for data pipelines
    out_clean = processed_dir / "cleaned.parquet"
    df.to_parquet(out_clean, index=False, engine='pyarrow')
    print(f"💾 Saved improved cleaned data → {out_clean}")

    return df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
# This allows the script to be:
# 1. Run directly from command line: `python src/etl.py`
# 2. Imported as a module: `from src.etl import run_etl`
#
# When run directly, __name__ == "__main__", so the code below executes
# When imported, __name__ == "etl", so the code below is skipped
if __name__ == "__main__":
    run_etl()
