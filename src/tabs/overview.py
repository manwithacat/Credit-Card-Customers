"""
Overview Tab - Dataset Summary and KPIs

This module renders the Overview tab showing high-level metrics,
data preview, and data type information.
"""

import streamlit as st
import pandas as pd


def render_overview_tab(filtered_df, churn_col, churn_colors=None):
    """
    Render the Overview tab with KPIs and data preview.

    This tab provides a high-level view of the dataset including:
    - Key metrics (customer count, features, churn rate)
    - Sample data preview
    - Data type information and missing value analysis

    Args:
        filtered_df (pd.DataFrame): The filtered dataset to display
        churn_col (str): Name of the churn column for metrics
        churn_colors (dict, optional): Color mapping for churn status values
    """
    st.header("Dataset Overview")

    # -------------------------------------------------------------------------
    # METRICS ROW - Display key numbers across 4 columns
    # -------------------------------------------------------------------------
    # st.columns(4) splits the page width into 4 equal columns
    # Think of it like a table row with 4 cells
    col1, col2, col3, col4 = st.columns(4)

    # Each "with colX:" puts content into that specific column
    with col1:
        # st.metric() displays a large number with a label - great for KPIs
        st.metric("Total Customers", f"{len(filtered_df):,}")

    with col2:
        # .columns gives us the number of columns (features) in our dataframe
        # Exclude naive_bayes columns from count
        feature_count = len([col for col in filtered_df.columns if 'naive_bayes' not in col.lower()])
        st.metric("Total Features", feature_count)

    with col3:
        if churn_col and churn_col in filtered_df.columns:
            # .str.contains() searches for text within string values
            # case=False means ignore capitalization
            # na=False means treat missing values as False
            # .sum() counts how many True values (rows with "Attrited")
            churn_count = (
                filtered_df[churn_col].str.contains("Attrited", case=False, na=False)
            ).sum()
            st.metric("Churned Customers", f"{churn_count:,}")
        else:
            st.metric("Churned Customers", "N/A")

    with col4:
        if churn_col and churn_col in filtered_df.columns:
            # .mean() gives proportion of True values (0 to 1)
            # * 100 converts to percentage
            # :.1f formats as decimal with 1 place after decimal point
            churn_rate = (
                filtered_df[churn_col].str.contains("Attrited", case=False, na=False)
            ).mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Churn Rate", "N/A")

    # -------------------------------------------------------------------------
    # DATA PREVIEW - Show first 100 rows
    # -------------------------------------------------------------------------
    st.subheader("Sample Data")
    # Filter out naive_bayes columns from display (synthetic features not useful for analysis)
    display_cols = [col for col in filtered_df.columns if 'naive_bayes' not in col.lower()]
    # st.dataframe() creates an interactive table that users can scroll and sort
    # .head(100) gets the first 100 rows
    # width="stretch" makes the table expand to fill available width
    st.dataframe(filtered_df[display_cols].head(100), width="stretch")

    # -------------------------------------------------------------------------
    # DATA TYPES TABLE - Show column info and missing values
    # -------------------------------------------------------------------------
    st.subheader("Data Types")
    # Filter out naive_bayes columns from data types display as well
    filtered_for_types = filtered_df[display_cols]
    # Create a new DataFrame that summarizes information about our data
    # This is a dictionary where keys become column names
    dtype_df = pd.DataFrame(
        {
            "Column": filtered_for_types.dtypes.index,  # Column names
            "Type": filtered_for_types.dtypes.values.astype(
                str
            ),  # Data types (int, float, object, etc.)
            "Non-Null": filtered_for_types.count().values,  # Number of non-missing values
            "Null %": (
                (1 - filtered_for_types.count() / len(filtered_for_types)) * 100
            )
            .round(2)
            .values,  # % missing
        }
    )
    st.dataframe(dtype_df, width="stretch")
