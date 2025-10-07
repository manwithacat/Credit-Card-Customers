import pandas as pd
from pathlib import Path

# Step 1: Extract
print("📥 Loading raw data...")
raw_path = Path("data/raw/BankChurners.csv")
df = pd.read_csv(raw_path)

print("✅ Raw data loaded:", df.shape)

# Step 2: Basic cleaning
print("🧹 Cleaning data...")
df.columns = df.columns.str.strip()  # remove spaces

# Drop unnecessary columns (like last two unnamed)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Map target variable
df["churn_flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

# Step 3: Save cleaned data
processed_path = Path("data/processed")
processed_path.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_path / "cleaned_basic.csv", index=False)

print("💾 Saved cleaned data to data/processed/cleaned_basic.csv")