import kagglehub
import shutil
import os
import pandas as pd
import json
import glob
import re
from pathlib import Path

# This script demonstrates how to work with Kaggle's API and manage datasets.
# It downloads the latest version of a dataset, organizes it into a local directory,
# and processes an accompanying data dictionary schema if available.
# This exercise shows practical usage of dataset ingestion and schema application,
# emphasizing how using a provided .xlsx schema can simplify later analysis by
# formalizing data structure, types, categories, and descriptions.

# Resolve project root two levels up from this file (…/da_bootcamp_p1)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define the dataset name to download; change this variable to pick a different dataset.
DATASET_NAME = "sakshigoyal7/credit-card-customers"

# Download latest version
path = kagglehub.dataset_download(DATASET_NAME)

# Copy the entire dataset directory into data/raw
destination = DATA_DIR / os.path.basename(path)
if destination.exists():
    shutil.rmtree(destination)
shutil.copytree(path, destination)


def parse_data_dictionary(dict_path: str, csv_columns: list[str] | None = None) -> dict:
    """
    Parse the provided Excel data dictionary to extract schema information.

    This function maximizes structure extraction from the dictionary, supporting
    data typing, categorical values, date parsing, and descriptive metadata.
    It flexibly identifies relevant columns and maps them to pandas-friendly types.

    If csv_columns is provided, it is used as a heuristic to identify the column name
    field when dictionary headers are non-standard or ambiguous.

    Args:
        dict_path (str): Path to the Excel file containing the data dictionary.
        csv_columns (list[str] | None): Optional list of column names from the CSV data,
            used to heuristically match the dictionary column containing variable names.

    Returns:
        dict: A schema dictionary with keys 'dtypes', 'parse_dates', 'categories',
              and 'descriptions' for use in downstream data processing.
    """
    df_dict = pd.read_excel(dict_path)
    # Normalize columns to lower-case and underscores
    df_dict.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df_dict.columns]

    # Expanded candidate lists
    col_name_candidates = [
        "column",
        "columns",
        "field",
        "name",
        "feature",
        "variable",
        "attribute",
        "column_name",
        "variable_name",
    ]
    type_candidates = ["type", "dtype", "data_type", "variable_type", "format"]
    allowed_candidates = [
        "allowed_values",
        "values",
        "categories",
        "allowed",
        "levels",
        "labels",
        "domain",
        "unique_values",
    ]
    desc_candidates = [
        "description",
        "desc",
        "definition",
        "meaning",
        "notes",
        "explanation",
    ]

    def find_col(candidates):
        # Try exact matches first
        for c in candidates:
            if c in df_dict.columns:
                return c
        # Then try substring matches (case-insensitive)
        for c in candidates:
            for col in df_dict.columns:
                if c.lower() in col.lower():
                    return col
        return None

    col_name_col = find_col(col_name_candidates)
    type_col = find_col(type_candidates)
    allowed_col = find_col(allowed_candidates)
    desc_col = find_col(desc_candidates)

    # Fallback heuristic if col_name_col still None and csv_columns provided
    if col_name_col is None and csv_columns is not None:
        best_col = None
        best_score = 0.0
        csv_set = set(s.strip() for s in csv_columns)
        for col in df_dict.columns:
            vals = df_dict[col].dropna().astype(str).str.strip()
            if len(vals) == 0:
                continue
            match_count = sum(v in csv_set for v in vals)
            score = match_count / len(vals)
            if score > best_score:
                best_score = score
                best_col = col
        if best_score >= 0.3:
            col_name_col = best_col

    if col_name_col is None:
        raise ValueError(
            f"Data dictionary is missing a column name field (expected one of: 'column','columns','field','name','feature','variable','attribute','column_name','variable_name'). Found columns: {df_dict.columns.tolist()}"
        )

    dtypes = {}
    parse_dates = []
    categories = {}
    descriptions = {}

    # Map string to pandas dtype
    def map_dtype(t):
        if pd.isna(t):
            return None
        t = str(t).strip().lower()
        if t in ["int", "integer"]:
            return "Int64"
        elif t in ["float", "double", "number"]:
            return "float64"
        elif t in ["string", "text", "object"]:
            return "string"
        elif t in ["bool", "boolean"]:
            return "boolean"
        elif t in ["date", "datetime", "time"]:
            return "date"
        elif t in ["category", "categorical"]:
            return "category"
        return None

    for _, row in df_dict.iterrows():
        if pd.isna(row.get(col_name_col)):
            continue
        col = str(row[col_name_col]).strip()
        dtype_raw = row.get(type_col) if type_col is not None else None
        dtype = map_dtype(dtype_raw)
        if dtype == "date":
            parse_dates.append(col)
            # No dtype set for dates, handled in parse_dates
        elif dtype is not None:
            dtypes[col] = dtype

        # Descriptions
        if desc_col is not None and pd.notna(row.get(desc_col)):
            descriptions[col] = str(row[desc_col]).strip()

        # Categories / allowed values
        if allowed_col is not None and pd.notna(row.get(allowed_col)):
            raw_vals = str(row[allowed_col])
            vals = []
            for part in raw_vals.split(","):
                part = part.strip()
                if ":" in part:
                    # key:value, take value part
                    parts = part.split(":", 1)
                    label = parts[1].strip()
                    vals.append(label)
                else:
                    vals.append(part)
            if vals:
                categories[col] = vals

    return {
        "dtypes": dtypes,
        "parse_dates": parse_dates,
        "categories": categories,
        "descriptions": descriptions,
    }


def apply_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Enforce the schema-defined data types, categories, and date parsing on the DataFrame.

    This function applies the parsed schema to the DataFrame, converting columns
    to appropriate pandas dtypes, setting categorical types with specified categories,
    and parsing date columns. It also attaches descriptions as DataFrame attributes.

    Args:
        df (pd.DataFrame): The DataFrame to which the schema will be applied.
        schema (dict): The schema dictionary containing dtypes, categories, parse_dates,
                       and descriptions.

    Returns:
        pd.DataFrame: A new DataFrame with schema applied.
    """
    df = df.copy()
    dtypes = schema.get("dtypes", {})
    categories = schema.get("categories", {})
    parse_dates = schema.get("parse_dates", [])

    # Apply dtypes
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "boolean":
                df[col] = df[col].astype("boolean")
            elif dtype == "Int64":
                df[col] = df[col].astype("Int64")
            elif dtype == "float64":
                df[col] = df[col].astype("float64")
            elif dtype == "category":
                # Will handle below
                pass
        except Exception:
            pass

    # Apply categories
    for col, cats in categories.items():
        if col in df.columns:
            try:
                cat_type = pd.api.types.CategoricalDtype(categories=cats, ordered=True)
                df[col] = df[col].astype(cat_type)
            except Exception:
                pass

    # Parse dates
    for col in parse_dates:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    df.attrs["dictionary"] = schema.get("descriptions", {})

    return df


def main():
    """
    Orchestrate the dataset ingestion, schema parsing, schema application, and reporting.

    This function demonstrates the full ingestion workflow:
    - Downloads and copies the dataset locally
    - Finds the CSV data and optional Excel dictionary schema
    - Parses the schema and applies it to the data
    - Prints a summary of the processing results
    - Saves the schema as JSON for reference

    This is an exercise to illustrate practical dataset management and schema usage,
    showing how formalizing data structure with a dictionary can aid downstream analysis.
    """
    # Find first CSV file in destination
    csv_files = glob.glob(str(destination / "**" / "*.csv"), recursive=True)
    csv_path = csv_files[0] if csv_files else None

    # Find first XLSX file (dictionary) in destination
    xlsx_files = glob.glob(str(destination / "**" / "*.xlsx"), recursive=True)
    xlsx_path = xlsx_files[0] if xlsx_files else None

    df = None
    schema = None
    if csv_path:
        df = pd.read_csv(csv_path)
        if xlsx_path:
            schema = parse_data_dictionary(xlsx_path, csv_columns=list(df.columns))
            df = apply_schema(df, schema)

    # Print summary
    print(f"Dataset has been copied to: {destination}")
    if schema is not None and df is not None:
        num_typed = len(schema.get("dtypes", {}))
        num_cats = len(schema.get("categories", {}))
        num_dates = len(schema.get("parse_dates", []))
        print(
            f"Dictionary found: typed {num_typed} columns, {num_cats} categorical, {num_dates} date columns"
        )
        print("DataFrame dtypes (first 10 columns):")
        print(df.dtypes.head(10))
    else:
        print("No dictionary found.")
        if df is not None:
            print("DataFrame dtypes (first 10 columns):")
            print(df.dtypes.head(10))

    # Save schema json if dictionary was found
    if schema:
        schema_out = DATA_DIR / (destination.name + "_schema.json")
        with open(schema_out, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    print("Dataset has been copied to:", destination)


if __name__ == "__main__":
    main()
