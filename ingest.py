# ============================================================================
# IMPORTS - External libraries we need
# ============================================================================
# kagglehub: Official Kaggle library for downloading datasets from Kaggle
import kagglehub

# shutil: Shell utilities - helps with file operations (copying, moving, deleting)
import shutil

# os: Operating system interface - file paths, environment variables, etc.
import os

# pandas (pd): The main data analysis library - for working with tables/DataFrames
import pandas as pd

# json: For reading and writing JSON (JavaScript Object Notation) files
import json

# glob: Pattern matching for file paths (e.g., find all *.csv files)
import glob

# re: Regular expressions - for pattern matching in strings
import re

# pathlib.Path: Modern, object-oriented way to work with file paths
from pathlib import Path

# ============================================================================
# SCRIPT OVERVIEW
# ============================================================================
# This script demonstrates how to work with Kaggle's API and manage datasets.
# It downloads the latest version of a dataset, organizes it into a local directory,
# and processes an accompanying data dictionary schema if available.
# This exercise shows practical usage of dataset ingestion and schema application,
# emphasizing how using a provided .xlsx schema can simplify later analysis by
# formalizing data structure, types, categories, and descriptions.

# ============================================================================
# SETUP - Define project directories
# ============================================================================
# __file__ is a special Python variable that contains the path to THIS script
# .resolve() makes it an absolute path (full path from root directory)
# .parent gets the directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

# Create path to data/raw directory using / operator
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Create the directory if it doesn't exist
# parents=True creates intermediate directories if needed (like 'mkdir -p')
# exist_ok=True means don't raise an error if directory already exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DOWNLOAD DATASET FROM KAGGLE
# ============================================================================
# Define the dataset name to download; change this variable to pick a different dataset.
# Format: "username/dataset-name" (you can find this in the Kaggle dataset URL)
DATASET_NAME = "sakshigoyal7/credit-card-customers"

# Download latest version of the dataset
# kagglehub automatically caches downloads, so repeated runs are fast
path = kagglehub.dataset_download(DATASET_NAME)

# ============================================================================
# COPY DATASET TO LOCAL PROJECT DIRECTORY
# ============================================================================
# os.path.basename() gets just the folder name from the full path
# Example: "/tmp/downloads/credit-card-customers" → "credit-card-customers"
destination = DATA_DIR / os.path.basename(path)

# If destination already exists, delete it first (clean slate)
if destination.exists():
    shutil.rmtree(destination)  # Remove directory and all its contents

# Copy the entire dataset directory to our project's data/raw folder
shutil.copytree(path, destination)


# ============================================================================
# FUNCTION: parse_data_dictionary
# ============================================================================
def parse_data_dictionary(dict_path: str, csv_columns: list[str] | None = None) -> dict:
    """
    Parse the provided Excel data dictionary to extract schema information.

    This function maximizes structure extraction from the dictionary, supporting
    data typing, categorical values, date parsing, and descriptive metadata.
    It flexibly identifies relevant columns and maps them to pandas-friendly types.

    If csv_columns is provided, it is used as a heuristic to identify the column name
    field when dictionary headers are non-standard or ambiguous.

    TYPE HINTS EXPLAINED:
    - dict_path: str → this parameter must be a string
    - csv_columns: list[str] | None = None → this parameter can be:
      * A list of strings, OR
      * None (the default value)
      The | means "or" (union type)
    - -> dict → this function returns a dictionary

    Args:
        dict_path (str): Path to the Excel file containing the data dictionary.
        csv_columns (list[str] | None): Optional list of column names from the CSV data,
            used to heuristically match the dictionary column containing variable names.

    Returns:
        dict: A schema dictionary with keys 'dtypes', 'parse_dates', 'categories',
              and 'descriptions' for use in downstream data processing.
    """
    # Read the Excel file into a pandas DataFrame
    df_dict = pd.read_excel(dict_path)

    # -------------------------------------------------------------------------
    # NORMALIZE COLUMN NAMES - Make them consistent and easy to work with
    # -------------------------------------------------------------------------
    # This is a LIST COMPREHENSION - it processes each column name in one line
    # For each column c:
    #   1. c.strip() removes leading/trailing whitespace
    #   2. .lower() converts to lowercase
    #   3. re.sub(r"\s+", "_", ...) replaces any whitespace with underscores
    # Example: "Column Name" → "column_name"
    df_dict.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df_dict.columns]

    # -------------------------------------------------------------------------
    # DEFINE CANDIDATE COLUMN NAMES
    # -------------------------------------------------------------------------
    # Different data dictionaries might use different column names
    # We define lists of possible names to search for
    # These are Python LISTS - ordered collections defined with square brackets []

    # Candidates for the column that contains variable/column names
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

    # Candidates for the column that describes data types
    type_candidates = ["type", "dtype", "data_type", "variable_type", "format"]

    # Candidates for the column that lists allowed/valid values
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

    # Candidates for the column that contains descriptions
    desc_candidates = [
        "description",
        "desc",
        "definition",
        "meaning",
        "notes",
        "explanation",
    ]

    # -------------------------------------------------------------------------
    # NESTED FUNCTION: find_col
    # -------------------------------------------------------------------------
    # This is a function INSIDE another function (nested function)
    # It has access to variables from the outer function (like df_dict)
    def find_col(candidates):
        """
        Search for a column in df_dict that matches one of the candidate names.

        First tries exact matches, then tries substring matches.

        Args:
            candidates: List of possible column names to search for

        Returns:
            The matching column name, or None if no match found
        """
        # Try exact matches first
        for c in candidates:
            if c in df_dict.columns:
                return c  # Found exact match, return immediately

        # Then try substring matches (case-insensitive)
        # Example: if candidate is "type", this matches column "data_type"
        for c in candidates:
            for col in df_dict.columns:
                if c.lower() in col.lower():
                    return col

        # If no matches found, return None
        return None

    # -------------------------------------------------------------------------
    # FIND COLUMNS IN DATA DICTIONARY
    # -------------------------------------------------------------------------
    # Call find_col() for each type of information we're looking for
    col_name_col = find_col(col_name_candidates)
    type_col = find_col(type_candidates)
    allowed_col = find_col(allowed_candidates)
    desc_col = find_col(desc_candidates)

    # -------------------------------------------------------------------------
    # FALLBACK HEURISTIC - If we couldn't find the column names column
    # -------------------------------------------------------------------------
    # If col_name_col is still None AND we have CSV columns to compare against,
    # try a more sophisticated approach: find which dictionary column has the most
    # overlap with actual CSV column names
    if col_name_col is None and csv_columns is not None:
        best_col = None      # Track which column is the best match
        best_score = 0.0     # Track how good the best match is (0-1)

        # Create a SET of CSV columns for fast lookup
        # SET is like a list but much faster for checking "is X in this collection?"
        # Generator expression: (s.strip() for s in csv_columns) processes each string
        csv_set = set(s.strip() for s in csv_columns)

        # Try each column in the data dictionary
        for col in df_dict.columns:
            # Get values from this column, removing missing values and whitespace
            vals = df_dict[col].dropna().astype(str).str.strip()

            if len(vals) == 0:
                continue  # Skip empty columns

            # Count how many values match CSV column names
            # sum() with a generator: sum(1 for each v that's in csv_set)
            match_count = sum(v in csv_set for v in vals)

            # Calculate match score (proportion of values that matched)
            score = match_count / len(vals)

            # Keep track of the best match so far
            if score > best_score:
                best_score = score
                best_col = col

        # If we found a column with at least 30% matches, use it
        if best_score >= 0.3:
            col_name_col = best_col

    # -------------------------------------------------------------------------
    # ERROR HANDLING - If we still can't find the column names column, raise error
    # -------------------------------------------------------------------------
    if col_name_col is None:
        # raise creates an error and stops the program
        # ValueError is a type of error - means a value is invalid/missing
        raise ValueError(
            f"Data dictionary is missing a column name field (expected one of: 'column','columns','field','name','feature','variable','attribute','column_name','variable_name'). Found columns: {df_dict.columns.tolist()}"
        )

    # -------------------------------------------------------------------------
    # INITIALIZE EMPTY COLLECTIONS - We'll fill these as we parse the dictionary
    # -------------------------------------------------------------------------
    # DICTIONARIES (key-value pairs, defined with curly braces {})
    dtypes = {}         # Will store {column_name: data_type}
    categories = {}     # Will store {column_name: [list, of, allowed, values]}
    descriptions = {}   # Will store {column_name: description_text}

    # LIST (ordered collection)
    parse_dates = []    # Will store [columns, that, should, be, dates]

    # -------------------------------------------------------------------------
    # NESTED FUNCTION: map_dtype
    # -------------------------------------------------------------------------
    # This function converts human-readable type names to pandas type names
    def map_dtype(t):
        """
        Map a string type description to a pandas dtype.

        Args:
            t: A string like "int", "float", "string", etc.

        Returns:
            A pandas dtype string, or None if unrecognized
        """
        # pd.isna() checks if value is missing (NaN, None, etc.)
        if pd.isna(t):
            return None

        # Normalize: convert to string, remove whitespace, make lowercase
        t = str(t).strip().lower()

        # Use if/elif/else chain to match type names
        # "in" checks if t is in the list
        if t in ["int", "integer"]:
            return "Int64"  # pandas nullable integer type
        elif t in ["float", "double", "number"]:
            return "float64"
        elif t in ["string", "text", "object"]:
            return "string"
        elif t in ["bool", "boolean"]:
            return "boolean"
        elif t in ["date", "datetime", "time"]:
            return "date"  # Special marker - we'll handle separately
        elif t in ["category", "categorical"]:
            return "category"

        # If no match, return None
        return None

    # -------------------------------------------------------------------------
    # ITERATE THROUGH DATA DICTIONARY ROWS - Build up our schema
    # -------------------------------------------------------------------------
    # .iterrows() loops through each row in the DataFrame
    # It returns (index, row) tuples - we use _ for index since we don't need it
    # _ is a Python convention for "I don't care about this value"
    for _, row in df_dict.iterrows():
        # Skip rows where the column name is missing
        # .get() safely retrieves a value, returns None if key doesn't exist
        if pd.isna(row.get(col_name_col)):
            continue  # Skip to next iteration

        # Get the column name from this row
        col = str(row[col_name_col]).strip()

        # -------------------------------------------------------------------------
        # PROCESS DATA TYPE
        # -------------------------------------------------------------------------
        # Get the raw type value (might be None if type_col wasn't found)
        dtype_raw = row.get(type_col) if type_col is not None else None
        # Map it to a pandas type
        dtype = map_dtype(dtype_raw)

        if dtype == "date":
            # Dates are handled specially - add to parse_dates list
            parse_dates.append(col)
            # Don't add to dtypes - pandas handles dates during reading
        elif dtype is not None:
            # For all other types, add to dtypes dictionary
            dtypes[col] = dtype

        # -------------------------------------------------------------------------
        # PROCESS DESCRIPTIONS
        # -------------------------------------------------------------------------
        # pd.notna() is opposite of pd.isna() - checks if value exists
        if desc_col is not None and pd.notna(row.get(desc_col)):
            descriptions[col] = str(row[desc_col]).strip()

        # -------------------------------------------------------------------------
        # PROCESS ALLOWED VALUES / CATEGORIES
        # -------------------------------------------------------------------------
        if allowed_col is not None and pd.notna(row.get(allowed_col)):
            # Get the raw values string (might be comma-separated)
            raw_vals = str(row[allowed_col])
            vals = []

            # Split by comma and process each part
            for part in raw_vals.split(","):
                part = part.strip()

                # Some dictionaries use "code:label" format, we want the label
                if ":" in part:
                    # Split on first colon only (.split(":", 1))
                    parts = part.split(":", 1)
                    label = parts[1].strip()  # Take the part after :
                    vals.append(label)
                else:
                    # No colon, use the whole value
                    vals.append(part)

            # Only add to categories if we found values
            if vals:
                categories[col] = vals

    # -------------------------------------------------------------------------
    # RETURN SCHEMA DICTIONARY
    # -------------------------------------------------------------------------
    # This dictionary contains all the schema information we extracted
    return {
        "dtypes": dtypes,
        "parse_dates": parse_dates,
        "categories": categories,
        "descriptions": descriptions,
    }


# ============================================================================
# FUNCTION: apply_schema
# ============================================================================
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
    # Create a copy so we don't modify the original DataFrame
    df = df.copy()

    # Extract schema components using .get() with default values
    # .get(key, default) returns the value if key exists, otherwise returns default
    dtypes = schema.get("dtypes", {})
    categories = schema.get("categories", {})
    parse_dates = schema.get("parse_dates", [])

    # -------------------------------------------------------------------------
    # APPLY DATA TYPES
    # -------------------------------------------------------------------------
    # .items() iterates through dictionary as (key, value) pairs
    for col, dtype in dtypes.items():
        # Skip if this column doesn't exist in our data
        if col not in df.columns:
            continue

        # TRY-EXCEPT block: try to do something, if it fails, don't crash
        try:
            # .astype() converts a column to a different data type
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "boolean":
                df[col] = df[col].astype("boolean")
            elif dtype == "Int64":
                df[col] = df[col].astype("Int64")
            elif dtype == "float64":
                df[col] = df[col].astype("float64")
            elif dtype == "category":
                # Categories are handled in the next section
                pass
        except Exception:
            # If conversion fails for any reason, just skip it
            # This makes the function more robust - it won't crash on bad data
            pass

    # -------------------------------------------------------------------------
    # APPLY CATEGORIES
    # -------------------------------------------------------------------------
    for col, cats in categories.items():
        if col in df.columns:
            try:
                # Create a categorical type with specific allowed values
                # ordered=True means categories have a meaningful order
                cat_type = pd.api.types.CategoricalDtype(categories=cats, ordered=True)
                df[col] = df[col].astype(cat_type)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # PARSE DATES
    # -------------------------------------------------------------------------
    for col in parse_dates:
        if col in df.columns:
            try:
                # pd.to_datetime() converts strings to datetime objects
                # errors="coerce" means invalid dates become NaT (Not a Time)
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # ATTACH DESCRIPTIONS AS METADATA
    # -------------------------------------------------------------------------
    # DataFrames have an .attrs dictionary for storing metadata
    # We store descriptions here so they travel with the DataFrame
    df.attrs["dictionary"] = schema.get("descriptions", {})

    return df


# ============================================================================
# FUNCTION: main
# ============================================================================
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
    # -------------------------------------------------------------------------
    # FIND DATA FILES
    # -------------------------------------------------------------------------
    # glob.glob() finds all files matching a pattern
    # ** means "any number of subdirectories" (recursive search)
    # *.csv means "any file ending in .csv"
    csv_files = glob.glob(str(destination / "**" / "*.csv"), recursive=True)
    # Get first CSV file if any exist, otherwise None
    csv_path = csv_files[0] if csv_files else None

    # Same pattern for Excel files (data dictionaries)
    xlsx_files = glob.glob(str(destination / "**" / "*.xlsx"), recursive=True)
    xlsx_path = xlsx_files[0] if xlsx_files else None

    # -------------------------------------------------------------------------
    # LOAD AND PROCESS DATA
    # -------------------------------------------------------------------------
    # Initialize variables as None
    df = None
    schema = None

    if csv_path:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        if xlsx_path:
            # If we have a data dictionary, parse it
            schema = parse_data_dictionary(xlsx_path, csv_columns=list(df.columns))
            # Apply the schema to enforce data types
            df = apply_schema(df, schema)

    # -------------------------------------------------------------------------
    # PRINT SUMMARY TO CONSOLE
    # -------------------------------------------------------------------------
    print(f"Dataset has been copied to: {destination}")

    if schema is not None and df is not None:
        # Count how many columns we found in each category
        num_typed = len(schema.get("dtypes", {}))
        num_cats = len(schema.get("categories", {}))
        num_dates = len(schema.get("parse_dates", []))

        print(
            f"Dictionary found: typed {num_typed} columns, {num_cats} categorical, {num_dates} date columns"
        )
        print("DataFrame dtypes (first 10 columns):")
        # .head(10) shows first 10 rows (or columns in this case)
        print(df.dtypes.head(10))
    else:
        print("No dictionary found.")
        if df is not None:
            print("DataFrame dtypes (first 10 columns):")
            print(df.dtypes.head(10))

    # -------------------------------------------------------------------------
    # SAVE SCHEMA TO JSON FILE
    # -------------------------------------------------------------------------
    if schema:
        # Create output file path
        schema_out = DATA_DIR / (destination.name + "_schema.json")

        # CONTEXT MANAGER: "with open(...) as f:" automatically closes the file when done
        # "w" means write mode, "encoding='utf-8'" handles special characters
        with open(schema_out, "w", encoding="utf-8") as f:
            # json.dump() writes Python dict to JSON file
            # indent=2 makes it pretty-printed (human-readable)
            json.dump(schema, f, indent=2)

    print("Dataset has been copied to:", destination)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
# This is a Python idiom that means "only run main() if this file is executed directly"
# If someone imports this file (import ingest), main() won't run automatically
# __name__ is a special variable:
#   - When running "python ingest.py": __name__ == "__main__"
#   - When importing "import ingest": __name__ == "ingest"
if __name__ == "__main__":
    main()
