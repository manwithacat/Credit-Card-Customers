"""
Distributions Tab - Feature Distribution Analysis

This module renders the Distributions tab showing how values are
distributed across numeric and categorical features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def render_distributions_tab(filtered_df, churn_col, churn_colors=None):
    """
    Render the Distributions tab with histograms and value counts.

    This tab provides distribution analysis including:
    - Interactive histograms for numeric features with adjustable bins
    - Summary statistics for numeric columns
    - Box plots for key financial features
    - Bar charts for categorical feature distributions

    Args:
        filtered_df (pd.DataFrame): The filtered dataset to analyze
        churn_col (str): Name of the churn column for color coding
        churn_colors (dict, optional): Color mapping for churn status values
    """
    st.header("Feature Distributions")

    # -------------------------------------------------------------------------
    # NUMERIC FEATURES SECTION
    # -------------------------------------------------------------------------
    # .select_dtypes() filters columns by their data type
    # include=[np.number] means "only numeric columns" (int, float, etc.)
    # .columns.tolist() converts the result to a regular Python list
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:  # Only show this section if we have numeric columns
        # Create 2 columns for user inputs side-by-side
        col1, col2 = st.columns(2)

        with col1:
            # st.selectbox() creates a dropdown menu (user can pick ONE option)
            # index=0 means the first item is selected by default
            selected_num = st.selectbox("Select Numeric Feature", numeric_cols, index=0)

        with col2:
            # st.slider() creates a slider widget for selecting a number
            # Arguments: label, min_value, max_value, default_value
            bins = st.slider("Number of Bins", 10, 100, 30)

        # -------------------------------------------------------------------------
        # HISTOGRAM - Show distribution of selected numeric feature
        # -------------------------------------------------------------------------
        # px.histogram() creates an interactive histogram chart
        # Plotly Express (px) makes it easy to create nice-looking charts
        fig = px.histogram(
            filtered_df,  # The data
            x=selected_num,  # Which column to plot
            nbins=bins,  # Number of bins in histogram (from slider above)
            title=f"Distribution of {selected_num}",
            marginal="box",  # Add a box plot on top
            color=churn_col if churn_col else None,  # Color bars by churn status if available
            color_discrete_map=churn_colors,  # Apply custom color scheme
        )
        # st.plotly_chart() displays the Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True, key="dist_numeric_hist")

        # -------------------------------------------------------------------------
        # SUMMARY STATISTICS TABLE
        # -------------------------------------------------------------------------
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)

        with col1:
            # Filter out ID columns and naive_bayes synthetic features from summary statistics
            stats_cols = [c for c in numeric_cols if 'clientnum' not in c.lower() and 'naive_bayes' not in c.lower()]

            # .describe() generates statistics: count, mean, std, min, quartiles, max
            stats_df = filtered_df[stats_cols].describe()

            # Format for appropriate precision based on feature type
            # Iterate through each column and apply appropriate rounding and formatting
            formatted_stats = stats_df.copy()
            for col in formatted_stats.columns:
                col_lower = col.lower()

                # Determine precision and format based on feature type
                if any(keyword in col_lower for keyword in ['score', 'ratio', 'utilization', 'rate']):
                    # Scores and ratios: 2 decimal places with commas
                    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:,.2f}")
                elif any(keyword in col_lower for keyword in ['amount', 'amt', 'limit', 'balance', 'revolving']):
                    # Dollar amounts: whole numbers with commas and USD symbol
                    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"${x:,.0f}")
                elif any(keyword in col_lower for keyword in ['count', 'ct', 'months', 'tenure', 'inactive', 'contacts']):
                    # Counts and time periods: whole numbers with commas
                    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:,.0f}")
                else:
                    # Default: 1 decimal place with commas
                    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:,.1f}")

            st.dataframe(formatted_stats, width="stretch")

        with col2:
            # Box plots for key financial features
            # This is a LIST COMPREHENSION - a Python shortcut for filtering lists
            # It reads: "include column c IF any keyword k is in the column name"
            # [:4] takes only the first 4 matches
            key_features = [
                c
                for c in numeric_cols
                if any(
                    k in c.lower()
                    for k in ["credit_limit", "balance", "transaction", "utilization"]
                )
            ][:4]

            if key_features:
                # .melt() transforms data from wide to long format
                # Wide: each feature is a column → Long: one "Feature" column, one "Value" column
                # This makes it easier to plot multiple features on one chart
                fig = px.box(
                    filtered_df[key_features].melt(
                        var_name="Feature", value_name="Value"
                    ),
                    x="Feature",
                    y="Value",
                    title="Key Features Box Plot",
                )
                # Rotate x-axis labels 45 degrees so they don't overlap
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="dist_key_features_box")

    # -------------------------------------------------------------------------
    # CATEGORICAL FEATURES SECTION
    # -------------------------------------------------------------------------
    st.subheader("Categorical Features")
    # Select non-numeric columns (text categories like "Male/Female", "Gold/Silver", etc.)
    cat_cols = filtered_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if cat_cols and len(cat_cols) > 0:
        selected_cat = st.selectbox("Select Categorical Feature", cat_cols)

        if selected_cat:
            # .value_counts() counts how many times each unique value appears
            # .head(15) keeps only the top 15 most common values
            value_counts = filtered_df[selected_cat].value_counts().head(15)

            # Create a bar chart
            fig = px.bar(
                x=value_counts.index,  # Category names
                y=value_counts.values,  # Counts
                title=f"Distribution of {selected_cat}",
                labels={"x": selected_cat, "y": "Count"},
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="dist_categorical_bar")
