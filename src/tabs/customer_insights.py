"""
Customer Insights Tab - Advanced Behavioral Pattern Analysis

This module renders the Customer Insights tab showing engineered features
and traditional behavioral patterns to understand customer churn drivers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def render_customer_insights_tab(filtered_df, churn_col, card_col):
    """
    Render the Customer Insights tab with engineered features and behavioral patterns.

    This tab provides advanced analytics including:
    - Engineered feature visualizations (risk scores, segmentation, RFM, engagement, lifecycle)
    - Traditional behavioral analysis (credit patterns, transaction behavior, utilization)
    - Multi-dimensional customer profiling
    - Segment performance summaries

    Args:
        filtered_df (pd.DataFrame): The filtered dataset to analyze
        churn_col (str): Name of the churn column
        card_col (str): Name of the card category column
    """
    st.header("Customer Insights & Behavioral Patterns")

    st.markdown("""
    Explore multi-dimensional customer behavior patterns to understand what drives churn.
    These visualizations showcase engineered features and derived metrics from the feature engineering pipeline.
    """)

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 1: Churn Risk Score
    # -------------------------------------------------------------------------
    st.subheader("⚠️ Churn Risk Score Analysis")

    if "churn_risk_score" in filtered_df.columns and "risk_category" in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Distribution of risk scores
            fig = px.histogram(
                filtered_df,
                x="churn_risk_score",
                color=churn_col if churn_col else None,
                barmode="overlay",
                title="Churn Risk Score Distribution",
                labels={"churn_risk_score": "Risk Score (0=Low, 1=High)"},
                nbins=30,
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk category breakdown
            if churn_col and churn_col in filtered_df.columns:
                risk_churn = pd.crosstab(
                    filtered_df["risk_category"],
                    filtered_df[churn_col],
                    normalize="index"
                ) * 100

                fig = px.bar(
                    risk_churn,
                    title="Actual Churn Rate by Risk Category",
                    labels={"value": "Churn Rate (%)", "index": "Risk Category"},
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: The engineered risk score combines transaction frequency, inactivity, engagement, and utilization patterns. Validate that high-risk customers actually churn more.")

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 2: Customer Segmentation
    # -------------------------------------------------------------------------
    st.subheader("🎯 Customer Value Segmentation")

    if "customer_segment" in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Segment distribution
            segment_counts = filtered_df["customer_segment"].value_counts()
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Segment Distribution",
                hole=0.4  # Creates a donut chart
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn rate by segment
            if churn_col and churn_col in filtered_df.columns:
                segment_churn = pd.crosstab(
                    filtered_df["customer_segment"],
                    filtered_df[churn_col],
                    normalize="index"
                ) * 100

                fig = px.bar(
                    segment_churn,
                    title="Churn Rate by Customer Segment",
                    labels={"value": "Churn Rate (%)", "index": "Customer Segment"},
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Segments combine RFM score and engagement. Champions (high RFM + engagement) should have lowest churn, while Hibernating customers are at highest risk.")

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 3: RFM Scores
    # -------------------------------------------------------------------------
    st.subheader("📊 RFM (Recency, Frequency, Monetary) Analysis")

    if all(col in filtered_df.columns for col in ["rfm_score", "frequency_score", "monetary_score"]):
        col1, col2 = st.columns(2)

        with col1:
            # RFM score vs churn
            fig = px.histogram(
                filtered_df,
                x="rfm_score",
                color=churn_col if churn_col else None,
                barmode="overlay",
                title="RFM Score Distribution by Churn Status",
                labels={"rfm_score": "RFM Score (0=Low, 1=High)"},
                nbins=20,
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Frequency vs Monetary scatter
            fig = px.scatter(
                filtered_df,
                x="frequency_score",
                y="monetary_score",
                color=churn_col if churn_col else None,
                title="Frequency vs Monetary Score",
                labels={
                    "frequency_score": "Frequency Score",
                    "monetary_score": "Monetary Score"
                },
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: RFM scores (0-1 normalized) show customer value. Low RFM scores indicate customers who transact infrequently and spend little - prime churn candidates.")

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 4: Engagement Score
    # -------------------------------------------------------------------------
    st.subheader("💪 Customer Engagement Score")

    if "engagement_score" in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Engagement distribution
            fig = px.histogram(
                filtered_df,
                x="engagement_score",
                color=churn_col if churn_col else None,
                barmode="overlay",
                title="Engagement Score Distribution",
                labels={"engagement_score": "Engagement Score (0=Low, 1=High)"},
                nbins=25,
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Engagement vs RFM (if available)
            if "rfm_score" in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x="engagement_score",
                    y="rfm_score",
                    color=churn_col if churn_col else None,
                    title="Engagement vs RFM Score",
                    labels={
                        "engagement_score": "Engagement Score",
                        "rfm_score": "RFM Score"
                    },
                    opacity=0.6
                )
                # Add quadrant lines to show segments
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Engagement combines products held, transaction frequency, and activity level. The quadrant plot shows how engagement and value intersect to create customer segments.")

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 5: Lifecycle Stage
    # -------------------------------------------------------------------------
    st.subheader("⏳ Customer Lifecycle Analysis")

    if "lifecycle_stage" in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Lifecycle distribution
            lifecycle_counts = filtered_df["lifecycle_stage"].value_counts()
            fig = px.bar(
                x=lifecycle_counts.index,
                y=lifecycle_counts.values,
                title="Customer Distribution by Lifecycle Stage",
                labels={"x": "Lifecycle Stage", "y": "Number of Customers"},
                color=lifecycle_counts.index
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn rate by lifecycle
            if churn_col and churn_col in filtered_df.columns:
                lifecycle_churn = pd.crosstab(
                    filtered_df["lifecycle_stage"],
                    filtered_df[churn_col],
                    normalize="index"
                ) * 100

                fig = px.bar(
                    lifecycle_churn,
                    title="Churn Rate by Lifecycle Stage",
                    labels={"value": "Churn Rate (%)", "index": "Lifecycle Stage"},
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Lifecycle stages segment customers by tenure (New/Growing/Mature/Loyal). Different stages may require different retention strategies.")

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 6: Transaction Behavior Metrics
    # -------------------------------------------------------------------------
    st.subheader("📈 Advanced Transaction Metrics")

    if "transaction_density" in filtered_df.columns or "monthly_spend_rate" in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            if "transaction_density" in filtered_df.columns:
                # Transaction density (transactions per month of tenure)
                fig = px.box(
                    filtered_df,
                    x=churn_col if churn_col else None,
                    y="transaction_density",
                    color=churn_col if churn_col else None,
                    title="Transaction Density by Churn Status",
                    labels={"transaction_density": "Transactions per Month"}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "monthly_spend_rate" in filtered_df.columns:
                # Monthly spend rate
                fig = px.box(
                    filtered_df,
                    x=churn_col if churn_col else None,
                    y="monthly_spend_rate",
                    color=churn_col if churn_col else None,
                    title="Monthly Spend Rate by Churn Status",
                    labels={"monthly_spend_rate": "$ Spent per Month"}
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: These normalized metrics account for customer tenure, making comparisons fairer between new and long-term customers.")

    # Add separator before original visualizations
    st.markdown("---")
    st.markdown("### Traditional Behavioral Analysis")
    st.caption("Original feature analysis using base data attributes:")

    # -------------------------------------------------------------------------
    # INSIGHT 1: Credit Limit vs Transaction Behavior
    # -------------------------------------------------------------------------
    st.subheader("💳 Credit Limit vs Transaction Activity")

    # Look for key financial columns
    credit_col = next((c for c in filtered_df.columns if 'credit_limit' in c.lower()), None)
    trans_amt_col = next((c for c in filtered_df.columns if 'total_trans_amt' in c.lower()), None)
    util_col = next((c for c in filtered_df.columns if 'utilization' in c.lower()), None)

    if credit_col and trans_amt_col:
        # Create scatter plot showing credit limit vs spending
        # Size of bubbles represents utilization if available
        fig = px.scatter(
            filtered_df,
            x=credit_col,
            y=trans_amt_col,
            color=churn_col if churn_col else None,
            size=util_col if util_col and util_col in filtered_df.columns else None,
            hover_data=[credit_col, trans_amt_col, util_col] if util_col else [credit_col, trans_amt_col],
            title="Credit Limit vs Total Transaction Amount (bubble size = utilization)",
            labels={credit_col: "Credit Limit", trans_amt_col: "Total Transactions ($)"},
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Look for clusters of high-limit but low-spending customers - these may be at risk of churn.")
    else:
        st.info("Credit limit or transaction amount data not available.")

    # -------------------------------------------------------------------------
    # INSIGHT 2: Transaction Frequency vs Average Transaction Size
    # -------------------------------------------------------------------------
    st.subheader("🔄 Transaction Patterns")

    trans_count_col = next((c for c in filtered_df.columns if 'total_trans_ct' in c.lower()), None)
    avg_trans_col = next((c for c in filtered_df.columns if 'avg_transaction' in c.lower()), None)

    if trans_count_col and avg_trans_col:
        col1, col2 = st.columns(2)

        with col1:
            # Scatter: frequency vs average size
            fig = px.scatter(
                filtered_df,
                x=trans_count_col,
                y=avg_trans_col,
                color=churn_col if churn_col else None,
                title="Transaction Frequency vs Avg Transaction Size",
                labels={trans_count_col: "Transaction Count", avg_trans_col: "Avg Transaction ($)"},
                opacity=0.6,
                trendline="ols" if len(filtered_df) > 10 else None
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Distribution comparison
            if churn_col and churn_col in filtered_df.columns:
                fig = px.box(
                    filtered_df,
                    x=churn_col,
                    y=trans_count_col,
                    color=churn_col,
                    title="Transaction Count by Churn Status",
                    labels={trans_count_col: "Transaction Count"}
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Customers with fewer transactions and lower amounts are more likely to churn.")
    else:
        st.info("Transaction pattern data not available.")

    # -------------------------------------------------------------------------
    # INSIGHT 3: Customer Engagement Score
    # -------------------------------------------------------------------------
    st.subheader("📊 Customer Engagement Analysis")

    # Look for relationship and engagement metrics
    rel_count_col = next((c for c in filtered_df.columns if 'relationship_count' in c.lower()), None)
    months_col = next((c for c in filtered_df.columns if 'months_on_book' in c.lower() or 'tenure' in c.lower()), None)
    inactive_col = next((c for c in filtered_df.columns if 'inactive' in c.lower()), None)

    if rel_count_col or months_col or inactive_col:
        col1, col2 = st.columns(2)

        with col1:
            # Relationship count distribution
            if rel_count_col and churn_col:
                # Create cross-tab for relationship count vs churn
                if rel_count_col in filtered_df.columns and churn_col in filtered_df.columns:
                    rel_churn = pd.crosstab(
                        filtered_df[rel_count_col],
                        filtered_df[churn_col],
                        normalize='index'
                    ) * 100

                    fig = px.bar(
                        rel_churn,
                        title="Churn Rate by Number of Products Held",
                        labels={'value': 'Percentage (%)', 'index': 'Number of Products'},
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("💡 **Insight**: Customers with more products tend to have lower churn rates (higher engagement).")

        with col2:
            # Tenure analysis
            if months_col and churn_col:
                fig = px.histogram(
                    filtered_df,
                    x=months_col,
                    color=churn_col,
                    barmode='overlay',
                    title="Customer Tenure Distribution",
                    labels={months_col: "Months on Book"},
                    nbins=30,
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 **Insight**: Compare tenure patterns between churned and active customers.")

    # -------------------------------------------------------------------------
    # INSIGHT 4: Card Utilization Patterns
    # -------------------------------------------------------------------------
    st.subheader("💰 Credit Utilization Patterns")

    if util_col and util_col in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Utilization distribution
            fig = px.histogram(
                filtered_df,
                x=util_col,
                color=churn_col if churn_col else None,
                barmode='overlay',
                title="Credit Utilization Distribution",
                labels={util_col: "Utilization Ratio"},
                nbins=40,
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Create utilization categories for analysis
            if not filtered_df[util_col].isna().all():
                # Create binned categories
                util_temp = filtered_df.copy()
                util_temp['util_category'] = pd.cut(
                    util_temp[util_col],
                    bins=[0, 0.3, 0.7, 1.0, float('inf')],
                    labels=['Low (0-30%)', 'Medium (30-70%)', 'High (70-100%)', 'Over 100%'],
                    include_lowest=True
                )

                if churn_col and churn_col in util_temp.columns:
                    # Churn rate by utilization category
                    util_churn = pd.crosstab(
                        util_temp['util_category'],
                        util_temp[churn_col],
                        normalize='index'
                    ) * 100

                    fig = px.bar(
                        util_churn,
                        title="Churn Rate by Credit Utilization Category",
                        labels={'value': 'Percentage (%)', 'index': 'Utilization Category'},
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Extreme utilization (very low or very high) may indicate churn risk.")
    else:
        st.info("Credit utilization data not available.")

    # -------------------------------------------------------------------------
    # INSIGHT 5: Customer Segmentation by Card Category
    # -------------------------------------------------------------------------
    st.subheader("🎯 Customer Segmentation by Card Type")

    if card_col and card_col in filtered_df.columns:
        # Get multiple metrics by card category
        card_metrics = filtered_df.groupby(card_col).agg({
            trans_amt_col: 'mean' if trans_amt_col and trans_amt_col in filtered_df.columns else lambda x: 0,
            credit_col: 'mean' if credit_col and credit_col in filtered_df.columns else lambda x: 0,
            trans_count_col: 'mean' if trans_count_col and trans_count_col in filtered_df.columns else lambda x: 0,
        }).reset_index()

        if trans_amt_col and trans_amt_col in filtered_df.columns:
            # Rename for clarity
            card_metrics.columns = [card_col, 'Avg Transaction Amount', 'Avg Credit Limit', 'Avg Transaction Count']

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    card_metrics,
                    x=card_col,
                    y='Avg Transaction Amount',
                    title="Average Transaction Amount by Card Category",
                    color=card_col
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    card_metrics,
                    x=card_col,
                    y='Avg Credit Limit',
                    title="Average Credit Limit by Card Category",
                    color=card_col
                )
                st.plotly_chart(fig, use_container_width=True)

            st.caption("💡 **Insight**: Different card tiers show distinct spending and credit patterns.")

    # -------------------------------------------------------------------------
    # INSIGHT 6: Multi-Dimensional Customer Profiles
    # -------------------------------------------------------------------------
    st.subheader("🔍 Multi-Dimensional Customer Analysis")

    st.markdown("""
    Explore how multiple customer attributes interact. Select dimensions to analyze:
    """)

    # Let users pick dimensions to explore
    col1, col2, col3 = st.columns(3)

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

    with col1:
        x_dim = st.selectbox(
            "X-axis (Numeric)",
            options=numeric_cols,
            index=0 if numeric_cols else 0,
            key='insights_x'
        )

    with col2:
        y_dim = st.selectbox(
            "Y-axis (Numeric)",
            options=numeric_cols,
            index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0,
            key='insights_y'
        )

    with col3:
        color_dim = st.selectbox(
            "Color By",
            options=[churn_col] + cat_cols if churn_col else cat_cols,
            index=0,
            key='insights_color'
        )

    if x_dim and y_dim and color_dim:
        fig = px.scatter(
            filtered_df,
            x=x_dim,
            y=y_dim,
            color=color_dim,
            title=f"Customer Segmentation: {x_dim} vs {y_dim}",
            opacity=0.6,
            hover_data=[x_dim, y_dim, color_dim]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight**: Identify customer segments and patterns that correlate with churn behavior.")

    # -------------------------------------------------------------------------
    # Summary Statistics by Segment
    # -------------------------------------------------------------------------
    st.subheader("📈 Segment Performance Summary")

    if churn_col and churn_col in filtered_df.columns:
        st.markdown("**Key Metrics Comparison: Active vs Churned Customers**")

        # Calculate summary statistics for key metrics
        summary_cols = []
        if trans_amt_col: summary_cols.append(trans_amt_col)
        if trans_count_col: summary_cols.append(trans_count_col)
        if credit_col: summary_cols.append(credit_col)
        if util_col: summary_cols.append(util_col)
        if avg_trans_col: summary_cols.append(avg_trans_col)

        if summary_cols:
            summary = filtered_df.groupby(churn_col)[summary_cols].agg(['mean', 'median', 'std'])
            st.dataframe(summary.round(2), width="stretch")

            st.caption("💡 **Insight**: Compare average behaviors between customer segments to identify churn indicators.")
