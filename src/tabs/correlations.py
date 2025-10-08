"""
Correlations Tab - Feature Relationship Analysis

This module renders the Correlations tab showing relationships
between numeric features through correlation analysis and scatter plots.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def render_correlations_tab(filtered_df, churn_col, churn_colors=None):
    """
    Render the Correlations tab with correlation heatmaps and scatter plots.

    This tab provides correlation analysis including:
    - Full correlation heatmap for all numeric features
    - Tables of strongest positive and negative correlations
    - Interactive scatter plots with trendlines
    - User-selectable feature pairs for exploration

    Args:
        filtered_df (pd.DataFrame): The filtered dataset to analyze
        churn_col (str): Name of the churn column for color coding
        churn_colors (dict, optional): Color mapping for churn status values
    """
    st.header("Correlation Analysis")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        # -------------------------------------------------------------------------
        # CORRELATION HEATMAP
        # -------------------------------------------------------------------------
        # .corr() calculates correlation coefficients between all pairs of numeric columns
        # Correlation ranges from -1 (perfect negative) to +1 (perfect positive)
        # 0 means no linear relationship
        corr_matrix = filtered_df[numeric_cols].corr()

        # Truncate long labels for better display, ensuring uniqueness
        def truncate_labels(labels, max_len=25):
            truncated = []
            seen = {}
            for label in labels:
                if len(label) <= max_len:
                    short = label
                else:
                    short = label[:max_len-3] + "..."

                # Handle duplicates by adding counter
                if short in seen:
                    seen[short] += 1
                    short = f"{short[:-3]}_{seen[short]}..."
                else:
                    seen[short] = 0

                truncated.append(short)
            return truncated

        short_labels = truncate_labels(corr_matrix.columns.tolist())

        # Create a copy with shortened labels
        corr_display = corr_matrix.copy()
        corr_display.columns = short_labels
        corr_display.index = short_labels

        # px.imshow() creates a heatmap (2D color-coded matrix)
        fig = px.imshow(
            corr_display,
            aspect="auto",  # Automatically adjust aspect ratio
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",  # Red-Blue colorscale (red=negative, blue=positive)
            zmin=-1,  # Minimum value for color scale
            zmax=1,  # Maximum value for color scale
        )
        # Show correlation values on cells, formatted to 2 decimals
        fig.update_traces(text=corr_matrix.round(2).values, texttemplate="%{text}")

        # Increase height and adjust layout for better visibility
        num_features = len(corr_matrix)
        height = max(600, num_features * 25)  # Dynamic height based on number of features
        fig.update_layout(
            height=height,
            xaxis={'side': 'bottom'},
            margin=dict(l=150, r=50, t=50, b=150)  # More margin for labels
        )
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

        # -------------------------------------------------------------------------
        # TOP CORRELATIONS TABLE
        # -------------------------------------------------------------------------
        st.subheader("Top Positive Correlations")

        # Extract upper triangle of correlation matrix to avoid duplicates
        # A correlation matrix is symmetric (corr(A,B) = corr(B,A))
        # We only need to look at pairs once
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(
                i + 1, len(corr_matrix.columns)
            ):  # j starts at i+1 (upper triangle)
                corr_pairs.append(
                    {
                        "Feature 1": corr_matrix.columns[i],
                        "Feature 2": corr_matrix.columns[j],
                        "Correlation": corr_matrix.iloc[i, j],
                    }
                )

        # Convert to DataFrame and sort by correlation strength
        corr_df = pd.DataFrame(corr_pairs).sort_values("Correlation", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Strongest Positive Correlations**")
            # .head(10) gets top 10 rows (strongest positive correlations)
            st.dataframe(corr_df.head(10), width="stretch")

        with col2:
            st.write("**Strongest Negative Correlations**")
            # .tail(10) gets bottom 10 rows (strongest negative correlations)
            st.dataframe(corr_df.tail(10), width="stretch")

        # -------------------------------------------------------------------------
        # SCATTER PLOT - Visualize relationship between two features
        # -------------------------------------------------------------------------
        st.subheader("Explore Feature Relationships")

        col1, col2 = st.columns(2)

        with col1:
            # Let user pick which features to compare
            x_feature = st.selectbox("X-axis", numeric_cols, index=0, key="x_axis")

        with col2:
            # Default to second column, or first if only one column exists
            # min() ensures we don't go out of bounds
            y_feature = st.selectbox(
                "Y-axis",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="y_axis",
            )

        # px.scatter() creates a scatter plot (each point is a customer)
        fig = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color=churn_col if churn_col else None,  # Color points by churn status
            color_discrete_map=churn_colors,  # Apply custom color scheme
            title=f"{x_feature} vs {y_feature}",
            trendline="ols"
            if len(filtered_df) > 10
            else None,  # Add linear regression line if enough data
            opacity=0.6,  # Make points slightly transparent so we can see overlaps
        )
        # Increase height for better visibility
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key="corr_scatter")

    else:
        st.warning("Not enough numeric columns for correlation analysis.")
