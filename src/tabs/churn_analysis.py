"""
Churn Analysis Tab - Understanding Customer Attrition

This module renders the Churn Analysis tab showing churn patterns
across different customer segments and features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def render_churn_analysis_tab(filtered_df, churn_col, card_col, churn_colors=None):
    """
    Render the Churn Analysis tab with segmented churn insights.

    This tab provides churn-focused analysis including:
    - Overall churn distribution pie chart
    - Churn rates by categorical features (card type, gender, etc.)
    - Churn distribution across numeric features (box plots)
    - Cross-tabulation analysis of churn by segments

    Args:
        filtered_df (pd.DataFrame): The filtered dataset to analyze
        churn_col (str): Name of the churn column
        card_col (str): Name of the card category column
        churn_colors (dict, optional): Color mapping for churn status values
    """
    st.header("Churn Analysis")

    if churn_col and churn_col in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # -------------------------------------------------------------------------
            # PIE CHART - Overall churn distribution
            # -------------------------------------------------------------------------
            churn_counts = filtered_df[churn_col].value_counts()
            # px.pie() creates a pie chart - good for showing proportions
            fig = px.pie(
                values=churn_counts.values,  # The sizes of pie slices
                names=churn_counts.index,  # The labels for each slice
                title="Churn Distribution",
                color=churn_counts.index,  # Color by churn status
                color_discrete_map=churn_colors,  # Apply custom color scheme
            )
            st.plotly_chart(fig, use_container_width=True, key="churn_pie_chart")

        with col2:
            # -------------------------------------------------------------------------
            # CHURN RATE BY CARD CATEGORY
            # -------------------------------------------------------------------------
            if card_col and card_col in filtered_df.columns:
                # pd.crosstab() creates a cross-tabulation (contingency table)
                # It counts how many rows fall into each combination of categories
                # normalize='index' converts counts to percentages (each row sums to 100%)
                churn_by_card = pd.crosstab(
                    filtered_df[card_col], filtered_df[churn_col], normalize="index"
                ) * 100  # Convert from 0-1 to 0-100

                fig = px.bar(
                    churn_by_card,
                    title=f"Churn Rate by {card_col}",
                    labels={"value": "Percentage (%)", "index": card_col},
                    barmode="group",  # Put bars side-by-side instead of stacked
                    color_discrete_map=churn_colors,  # Apply custom color scheme
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="churn_by_card_chart")

        # -------------------------------------------------------------------------
        # CHURN BY NUMERIC FEATURES - Compare distributions between churned/not churned
        # -------------------------------------------------------------------------
        st.subheader("Churn by Numeric Features")

        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        # List comprehension: filter to key financial columns, take first 6
        key_numeric = [
            c
            for c in numeric_cols
            if any(
                k in c.lower()
                for k in [
                    "credit_limit",
                    "balance",
                    "transaction",
                    "utilization",
                    "count",
                    "amount",
                ]
            )
        ][:6]

        if key_numeric:
            col1, col2 = st.columns(2)

            # Loop through features and create box plots
            # enumerate() gives us both index (idx) and value (feature) while looping
            for idx, feature in enumerate(key_numeric):
                # Put even-indexed items (0,2,4...) in col1, odd-indexed (1,3,5...) in col2
                # % is the modulo operator: idx % 2 gives remainder when dividing by 2
                with col1 if idx % 2 == 0 else col2:
                    # Box plot shows median, quartiles, and outliers for each group
                    fig = px.box(
                        filtered_df,
                        x=churn_col,
                        y=feature,
                        title=f"{feature} by Churn Status",
                        color=churn_col,
                        color_discrete_map=churn_colors,  # Apply custom color scheme
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"churn_box_{feature}")

        # -------------------------------------------------------------------------
        # CHURN BY CATEGORICAL FEATURES
        # -------------------------------------------------------------------------
        st.subheader("Churn by Categorical Features")

        # Get all categorical columns except the churn column itself
        cat_cols = [
            c
            for c in filtered_df.select_dtypes(
                include=["object", "category"]
            ).columns
            if c != churn_col
        ]

        if cat_cols:
            # key='churn_cat' gives this widget a unique ID so Streamlit can track it
            # This prevents conflicts if we have another selectbox with same label elsewhere
            selected_cat = st.selectbox(
                "Select Feature for Churn Analysis", cat_cols, key="churn_cat"
            )

            if selected_cat:
                # Same crosstab pattern as before - show churn % for each category
                churn_by_cat = pd.crosstab(
                    filtered_df[selected_cat], filtered_df[churn_col], normalize="index"
                ) * 100

                fig = px.bar(
                    churn_by_cat,
                    title=f"Churn Rate by {selected_cat}",
                    labels={"value": "Percentage (%)", "index": selected_cat},
                    barmode="group",
                    color_discrete_map=churn_colors,  # Apply custom color scheme
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="churn_by_categorical")

    else:
        # st.warning() shows a yellow warning message
        st.warning(
            "No churn column found in the dataset. Expected column name: 'Attrition_Flag'"
        )
