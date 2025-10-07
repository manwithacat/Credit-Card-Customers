import pandas as pd
from pathlib import Path

raw_path = Path("data/raw/BankChurners.csv")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

print("📥 Loading raw data...")
df = pd.read_csv(raw_path)
print("✅ Raw data loaded:", df.shape)

print("🧹 Cleaning data...")
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
    .str.replace("/", "_")
    .str.lower()
)

for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

if "attrition_flag" in df.columns:
    df["churn_flag"] = df["attrition_flag"].map({
        "Attrited Customer": 1,
        "Existing Customer": 0
    }).astype("Int64")

for col in ["education_level", "income_category", "marital_status"]:
    if col in df.columns:
        df[col] = df[col].replace({"Unknown": pd.NA, "unknown": pd.NA})

int_like = [
    "dependent_count","months_on_book","months_inactive_12_mon",
    "contacts_count_12_mon","total_relationship_count","total_trans_ct"
]
for c in int_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

float_like = [
    "total_trans_amt","credit_limit","total_revolving_bal","avg_open_to_buy",
    "total_amt_chng_q4_q1","total_ct_chng_q4_q1","avg_utilization_ratio"
]
for c in float_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if {"total_trans_amt","total_trans_ct"}.issubset(df.columns):
    df["avg_transaction"] = df["total_trans_amt"] / df["total_trans_ct"].replace(0, pd.NA)

if "months_on_book" in df.columns:
    df["tenure_years"] = (df["months_on_book"].astype("float") / 12).round(2)

if "avg_utilization_ratio" in df.columns:
    df["utilization"] = df["avg_utilization_ratio"]
elif {"avg_open_to_buy","total_revolving_bal"}.issubset(df.columns):
    denom = df["avg_open_to_buy"] + df["total_revolving_bal"]
    df["utilization"] = df["total_revolving_bal"] / denom.replace(0, pd.NA)

before = len(df)
df = df.drop_duplicates()
removed = before - len(df)
print(f"🧽 Duplicates removed: {removed}")

out_clean = processed_dir / "cleaned.csv"
df.to_csv(out_clean, index=False)
print(f"💾 Saved improved cleaned data → {out_clean}")