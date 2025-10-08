"""
Credit Card Customer Churn Analysis Dashboard

A simple Streamlit app that visualizes credit card customer data
and basic churn patterns from the interim parquet snapshot.
"""

# ============================================================================
# IMPORTS - Libraries we need for this application
# ============================================================================
# pathlib: Used for working with file paths in a cross-platform way (works on Windows, Mac, Linux)
import pathlib

# pandas (pd): The main library for working with tabular data (like Excel spreadsheets)
import pandas as pd

# numpy (np): Library for numerical operations and working with arrays
import numpy as np

# streamlit (st): The web app framework - turns Python scripts into interactive web apps!
# Everything that starts with "st." is a Streamlit function that displays something
import streamlit as st

# plotly: Libraries for creating interactive charts and graphs
import plotly.express as px  # px = simple, high-level plotting functions
import plotly.graph_objects as go  # go = more detailed, customizable plotting

# seaborn & matplotlib: Alternative plotting libraries (not heavily used in this app)
import seaborn as sns
import matplotlib.pyplot as plt

# Tab modules: Modular rendering functions for each dashboard tab
from src.tabs import (
    render_overview_tab,
    render_distributions_tab,
    render_churn_analysis_tab,
    render_correlations_tab,
    render_customer_insights_tab,
    render_churn_comparison_tab,
)

# ============================================================================
# COLOR SCHEME CONFIGURATION
# ============================================================================
# Semantic color mapping for churn status visualization
# Blue/Orange scheme chosen for color-blind accessibility
CHURN_COLORS = {
    "Existing Customer": "#3498db",  # Blue - retained customer (calm, stable)
    "Attrited Customer": "#e67e22"   # Orange - churned customer (warning, attention)
}


# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================
@st.cache_data  # This is a "decorator" - it tells Streamlit to cache (save) the result
                # so we don't reload the data every time the user interacts with the app.
                # This makes the app MUCH faster! The data only loads once.
def load_data():
    """
    Load the feature-engineered parquet file, running ETL/feature pipelines if needed.

    This function looks for our feature-engineered data file created by src/features.py.
    If the parquet file doesn't exist, it automatically runs both the ETL pipeline
    and feature engineering pipeline to generate it.

    Pipeline: raw CSV → ETL (cleaned.parquet) → Features (features.parquet) → Dashboard

    This ensures the dashboard always has enriched data to display without requiring
    users to manually run the pipeline scripts first.

    Returns:
        pd.DataFrame: A pandas DataFrame containing our customer data with engineered features
    """
    # __file__ is a special variable that contains the path to THIS script (app.py)
    # .resolve().parent gets the directory that contains this script
    root = pathlib.Path(__file__).resolve().parent

    # The / operator with Path objects joins paths together
    # This is like "root/data/processed/features.parquet"
    features_file = root / "data" / "processed" / "features.parquet"
    cleaned_file = root / "data" / "processed" / "cleaned.parquet"

    # Check if the feature-engineered parquet exists
    if not features_file.exists():
        st.info("🔄 Feature-engineered data not found. Running pipeline...")

        # First, ensure we have cleaned data
        if not cleaned_file.exists():
            # Need to run ETL first
            try:
                import sys
                sys.path.insert(0, str(root))  # Add project root to Python path
                from src.etl import run_etl

                st.info("   → Running ETL pipeline...")
                run_etl()
                st.success("   ✅ ETL pipeline completed!")
            except Exception as e:
                st.warning(f"⚠️ ETL pipeline failed: {e}")
                # Try to fall back to raw CSV
                raw = root / "data" / "raw"
                try:
                    csv_file = next(raw.rglob("*.csv"))
                    df = pd.read_csv(csv_file)
                    df = df.rename(columns={
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'naive_bayes_1',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'naive_bayes_2'
                    })
                    return df
                except StopIteration:
                    return pd.DataFrame()

        # Now run feature engineering
        try:
            import sys
            sys.path.insert(0, str(root))
            from src.features import run_feature_engineering

            st.info("   → Running feature engineering...")
            run_feature_engineering()
            st.success("✅ Feature engineering completed!")
        except Exception as e:
            st.warning(f"⚠️ Feature engineering failed: {e}")
            # Fall back to cleaned data if available
            if cleaned_file.exists():
                st.info("   → Loading cleaned data without features as fallback")
                df = pd.read_parquet(cleaned_file)
                df = df.rename(columns={
                    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'naive_bayes_1',
                    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'naive_bayes_2'
                })
                return df
            return pd.DataFrame()

    # Try to load the feature-engineered parquet file
    if features_file.exists():
        # Parquet is a fast, compressed file format for data (better than CSV)
        df = pd.read_parquet(features_file)
        # Rename excessively long naive bayes column names
        df = df.rename(columns={
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'naive_bayes_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'naive_bayes_2'
        })
        return df

    # Final fallback to cleaned data
    if cleaned_file.exists():
        st.warning("⚠️ Loading cleaned data without engineered features.")
        df = pd.read_parquet(cleaned_file)
        df = df.rename(columns={
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'naive_bayes_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'naive_bayes_2'
        })
        return df

    # Last resort: raw CSV
    st.warning("⚠️ Loading raw CSV as fallback. Data may not be fully processed.")
    raw = root / "data" / "raw"
    try:
        csv_file = next(raw.rglob("*.csv"))
        df = pd.read_csv(csv_file)
        df = df.rename(columns={
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'naive_bayes_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'naive_bayes_2'
        })
        return df
    except StopIteration:
        return pd.DataFrame()


# ============================================================================
# PAGE CONFIGURATION - This MUST be the first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="Credit Card Churn Analysis",  # Text shown in browser tab
    page_icon="💳",  # Emoji shown in browser tab
    layout="wide",  # Use full width of browser (vs "centered")
    initial_sidebar_state="expanded"  # Sidebar open by default when page loads
)

# ============================================================================
# LOAD DATA - Call our function to load the data
# ============================================================================
# This runs the load_data() function we defined above
# Because of @st.cache_data, this only actually loads once, then is reused
df = load_data()

# Check if DataFrame is empty (no data found)
if df.empty:
    # st.error() displays a red error message to the user
    st.error("⚠️ No data found. Please run src/etl.py and src/features.py to generate the processed data.")
    # st.stop() halts execution here - nothing below this line will run
    st.stop()

# ============================================================================
# HEADER SECTION - Title and description
# ============================================================================
# st.title() creates a large heading at the top of the page
st.title("💳 Credit Card Customer Churn Analysis")

# st.markdown() lets us write formatted text using Markdown syntax
# The triple quotes (""") allow multi-line strings in Python
st.markdown("""
Interactive dashboard for exploring credit card customer data and churn patterns.
Data loaded from feature-engineered parquet (ETL → Feature Engineering → Dashboard).
""")

# ============================================================================
# SIDEBAR - Navigation and filters
# ============================================================================
# st.sidebar lets us put widgets in the left sidebar instead of main page
# This helps organize our app and keeps filters accessible
st.sidebar.header("Filters")

# ============================================================================
# DETECT KEY COLUMNS - Find columns dynamically by name
# ============================================================================
# We don't hardcode column names because different datasets might use different names
# This makes our code more flexible and reusable

# The next() function with a generator expression is a Python pattern for finding the first match
# Generator expression: (c for c in df.columns if c.lower() in {"attrition_flag", "churn", "churned"})
# - This loops through all column names in df.columns
# - For each column c, it checks if the lowercase version matches our target names
# - If match found, next() returns it; if no match, returns None (the second argument)
churn_col = next(
    (c for c in df.columns if c.lower() in {"attrition_flag", "churn", "churned"}),
    None  # Default value if no match found
)

# Same pattern for other categorical columns we want to filter by
gender_col = next((c for c in df.columns if c.lower() in {"gender"}), None)
education_col = next(
    (c for c in df.columns if c.lower() in {"education_level", "education"}), None
)
card_col = next((c for c in df.columns if c.lower() in {"card_category"}), None)

# ============================================================================
# CREATE FILTERED DATASET - Start with a copy of original data
# ============================================================================
# .copy() creates a duplicate so we don't modify the original df
# We'll apply filters to this copy based on user selections
filtered_df = df.copy()

# ============================================================================
# FILTER WIDGETS - Create interactive filters in the sidebar
# ============================================================================
# Each filter follows the same pattern:
# 1. Check if the column exists (if churn_col and churn_col in df.columns)
# 2. Create a multiselect widget for user to choose values
# 3. Filter the dataframe to only include rows where the column matches selected values

# CHURN STATUS FILTER
if churn_col and churn_col in df.columns:
    # st.sidebar.multiselect() creates a dropdown where users can select multiple options
    churn_filter = st.sidebar.multiselect(
        "Churn Status",  # Label shown to user
        options=df[churn_col].unique(),  # .unique() gets all distinct values in the column
        default=df[churn_col].unique()   # Start with all options selected
    )
    # Filter the dataframe: keep only rows where churn_col value is in the user's selection
    # .isin() checks if each row's value is in the churn_filter list
    filtered_df = filtered_df[filtered_df[churn_col].isin(churn_filter)]

# GENDER FILTER
if gender_col and gender_col in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Gender",
        # .dropna() removes any null/missing values before getting unique values
        options=df[gender_col].dropna().unique(),
        default=df[gender_col].dropna().unique()
    )
    filtered_df = filtered_df[filtered_df[gender_col].isin(gender_filter)]

# CARD CATEGORY FILTER
if card_col and card_col in df.columns:
    card_filter = st.sidebar.multiselect(
        "Card Category",
        options=df[card_col].dropna().unique(),
        default=df[card_col].dropna().unique()
    )
    filtered_df = filtered_df[filtered_df[card_col].isin(card_filter)]

# Show how many records remain after filtering
# f"string {variable}" is an f-string - it inserts variable values into the string
# :, formats numbers with thousands separators (e.g., 1,000 instead of 1000)
st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")

# ============================================================================
# MAIN CONTENT - Organized into tabs for better UX
# ============================================================================
# st.tabs() creates a tabbed interface - like browser tabs but inside our app
# Users can click between tabs to see different views of the data
# This returns 6 tab objects that we assign to tab1, tab2, tab3, tab4, tab5, tab6
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Distributions",
    "🔍 Churn Analysis",
    "🔬 Churn Comparison",
    "🔗 Correlations",
    "💡 Customer Insights"
])

# ============================================================================
# TAB 1: OVERVIEW - High-level metrics and data preview
# ============================================================================
with tab1:
    render_overview_tab(filtered_df, churn_col, churn_colors=CHURN_COLORS)

# ============================================================================
# TAB 2: DISTRIBUTIONS - Explore how values are spread across features
# ============================================================================
with tab2:
    render_distributions_tab(filtered_df, churn_col, churn_colors=CHURN_COLORS)

# ============================================================================
# TAB 3: CHURN ANALYSIS - Understand who churned and why
# ============================================================================
with tab3:
    render_churn_analysis_tab(filtered_df, churn_col, card_col, churn_colors=CHURN_COLORS)

# ============================================================================
# TAB 4: CHURN COMPARISON - Direct comparison ignoring churn filter
# ============================================================================
with tab4:
    # Pass both full dataset (df) and filtered dataset (filtered_df)
    # This tab ignores churn status filter but respects other filters
    render_churn_comparison_tab(df, filtered_df, churn_col, churn_colors=CHURN_COLORS)

# ============================================================================
# TAB 5: CORRELATIONS - Find relationships between numeric features
# ============================================================================
with tab5:
    render_correlations_tab(filtered_df, churn_col, churn_colors=CHURN_COLORS)

# ============================================================================
# TAB 6: CUSTOMER INSIGHTS - Advanced behavioral pattern analysis
# ============================================================================
with tab6:
    render_customer_insights_tab(filtered_df, churn_col, card_col, churn_colors=CHURN_COLORS)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")  # Horizontal line
st.markdown("""
**Credit Card Customer Churn Analysis Dashboard**
Built with Streamlit • Data from Kaggle
""")
