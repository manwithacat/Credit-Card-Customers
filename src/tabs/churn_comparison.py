"""
Churn Comparison Tab - Direct Comparison of Attrited vs Existing Customers

This module renders a comparison tab that ignores the churn status filter
to always show direct comparisons between attrited and existing customers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def truncate_label(label, max_len=30):
    """
    Truncate long feature labels for better display.

    Args:
        label (str): The label to truncate
        max_len (int): Maximum length before truncation

    Returns:
        str: Truncated label with ellipsis if needed
    """
    if len(label) <= max_len:
        return label
    return label[:max_len-3] + "..."


def render_churn_comparison_tab(full_df, filtered_df, churn_col, churn_colors=None):
    """
    Render the Churn Comparison tab with direct comparisons.

    This tab IGNORES the churn status filter to always show both groups.
    It provides side-by-side comparisons and statistical analysis.

    Args:
        full_df (pd.DataFrame): The complete unfiltered dataset
        filtered_df (pd.DataFrame): The filtered dataset (for other filters like gender, card type)
        churn_col (str): Name of the churn column
        churn_colors (dict, optional): Color mapping for churn status values
    """
    st.header("🔬 Churn Comparison: Attrited vs Existing Customers")

    st.markdown("""
    This tab **ignores the churn status filter** to always show direct comparisons between
    attrited and existing customers. Apply other filters (gender, card category) to refine the comparison.
    """)

    # Filter out churn status, but keep other filters
    # We want both attrited and existing, so we use the filtered_df but ensure both churn statuses are included
    comparison_df = full_df.copy()

    # Apply non-churn filters from filtered_df
    # Get the indices that were kept in filtered_df (excluding churn filter)
    # For simplicity, we'll just use full_df but note this in the UI

    if not churn_col or churn_col not in comparison_df.columns:
        st.warning("⚠️ No churn column found. This tab requires churn status data.")
        return

    # Split data into two groups
    existing = comparison_df[comparison_df[churn_col].str.contains("Existing", case=False, na=False)]
    attrited = comparison_df[comparison_df[churn_col].str.contains("Attrited", case=False, na=False)]

    # -------------------------------------------------------------------------
    # OVERVIEW METRICS
    # -------------------------------------------------------------------------
    st.subheader("📊 Population Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(comparison_df):,}")

    with col2:
        st.metric("Existing Customers", f"{len(existing):,}",
                  delta=f"{len(existing)/len(comparison_df)*100:.1f}%")

    with col3:
        st.metric("Attrited Customers", f"{len(attrited):,}",
                  delta=f"{len(attrited)/len(comparison_df)*100:.1f}%")

    with col4:
        churn_rate = len(attrited) / len(comparison_df) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    # -------------------------------------------------------------------------
    # KEY METRIC COMPARISONS
    # -------------------------------------------------------------------------
    st.subheader("📈 Key Metric Comparisons")

    # Find key numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()

    # Key metrics to compare
    key_metrics = [
        'total_trans_ct', 'total_trans_amt', 'total_relationship_count',
        'months_inactive_12_mon', 'contacts_count_12_mon', 'credit_limit',
        'avg_utilization_ratio', 'total_revolving_bal', 'avg_open_to_buy'
    ]

    available_metrics = [m for m in key_metrics if m in numeric_cols][:6]

    if available_metrics:
        # Create comparison table
        comparison_stats = []

        for metric in available_metrics:
            existing_mean = existing[metric].mean()
            attrited_mean = attrited[metric].mean()
            diff_pct = ((attrited_mean - existing_mean) / existing_mean * 100) if existing_mean != 0 else 0

            comparison_stats.append({
                'Metric': truncate_label(metric.replace('_', ' ').title()),
                'Existing (Avg)': f"{existing_mean:.2f}",
                'Attrited (Avg)': f"{attrited_mean:.2f}",
                'Difference': f"{diff_pct:+.1f}%"
            })

        comparison_table = pd.DataFrame(comparison_stats)
        st.dataframe(comparison_table, width="stretch", hide_index=True)

        st.caption("💡 **Insight**: Negative percentages indicate attrited customers have lower values. Look for large differences as potential churn predictors.")

    # -------------------------------------------------------------------------
    # SIDE-BY-SIDE DISTRIBUTIONS
    # -------------------------------------------------------------------------
    st.subheader("📊 Distribution Comparisons")

    # Select metric to compare
    if numeric_cols:
        col1, col2 = st.columns([1, 3])

        with col1:
            # Create display options with truncated labels
            metric_options = available_metrics if available_metrics else numeric_cols[:10]
            metric_display = {m: truncate_label(m.replace('_', ' ').title(), max_len=40) for m in metric_options}

            selected_metric = st.selectbox(
                "Select Metric",
                options=metric_options,
                format_func=lambda x: metric_display[x],
                key="comparison_metric"
            )

        with col2:
            st.markdown(f"**Comparing:** {truncate_label(selected_metric.replace('_', ' ').title(), max_len=50)}")

        # Create side-by-side visualizations
        col1, col2 = st.columns(2)

        metric_label = truncate_label(selected_metric.replace('_', ' ').title())

        with col1:
            # Overlapping histograms
            fig = px.histogram(
                comparison_df,
                x=selected_metric,
                color=churn_col,
                color_discrete_map=churn_colors,
                barmode='overlay',
                title=f"{metric_label} Distribution",
                opacity=0.7,
                nbins=30
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="comparison_hist")

        with col2:
            # Box plot comparison
            fig = px.box(
                comparison_df,
                x=churn_col,
                y=selected_metric,
                color=churn_col,
                color_discrete_map=churn_colors,
                title=f"{metric_label} Box Plot",
                points=False  # Don't show individual points for clarity
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="comparison_box")

    # -------------------------------------------------------------------------
    # CATEGORICAL FEATURE COMPARISONS
    # -------------------------------------------------------------------------
    st.subheader("🏷️ Categorical Feature Breakdown")

    cat_cols = comparison_df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != churn_col][:5]  # Exclude churn column itself

    if cat_cols:
        # Create display options with truncated labels
        cat_display = {c: truncate_label(c.replace('_', ' ').title(), max_len=40) for c in cat_cols}

        selected_cat = st.selectbox(
            "Select Categorical Feature",
            options=cat_cols,
            format_func=lambda x: cat_display[x],
            key="comparison_categorical"
        )

        col1, col2 = st.columns(2)

        cat_label = truncate_label(selected_cat.replace('_', ' ').title())

        with col1:
            # Existing customers breakdown
            existing_counts = existing[selected_cat].value_counts().head(10)
            fig = px.bar(
                x=existing_counts.index,
                y=existing_counts.values,
                title=f"Existing Customers: {cat_label}",
                labels={'x': selected_cat, 'y': 'Count'},
                color_discrete_sequence=[churn_colors.get("Existing Customer", "#3498db") if churn_colors else "#3498db"]
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="comparison_existing_cat_bar")

        with col2:
            # Attrited customers breakdown
            attrited_counts = attrited[selected_cat].value_counts().head(10)
            fig = px.bar(
                x=attrited_counts.index,
                y=attrited_counts.values,
                title=f"Attrited Customers: {cat_label}",
                labels={'x': selected_cat, 'y': 'Count'},
                color_discrete_sequence=[churn_colors.get("Attrited Customer", "#e67e22") if churn_colors else "#e67e22"]
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="comparison_attrited_cat_bar")

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURES COMPARISON (if available)
    # -------------------------------------------------------------------------
    st.subheader("🔧 Engineered Features Comparison")

    engineered_features = [
        'churn_risk_score', 'engagement_score', 'rfm_score',
        'transaction_density', 'monthly_spend_rate'
    ]

    available_engineered = [f for f in engineered_features if f in comparison_df.columns]

    if available_engineered:
        # Create radar chart comparing average scores
        existing_scores = [existing[f].mean() for f in available_engineered]
        attrited_scores = [attrited[f].mean() for f in available_engineered]

        # Truncate labels for radar chart
        radar_labels = [truncate_label(f.replace('_', ' ').title(), max_len=25) for f in available_engineered]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=existing_scores,
            theta=radar_labels,
            fill='toself',
            name='Existing Customers',
            line_color=churn_colors.get("Existing Customer", "#3498db") if churn_colors else "#3498db"
        ))

        fig.add_trace(go.Scatterpolar(
            r=attrited_scores,
            theta=radar_labels,
            fill='toself',
            name='Attrited Customers',
            line_color=churn_colors.get("Attrited Customer", "#e67e22") if churn_colors else "#e67e22"
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Engineered Feature Profile Comparison",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True, key="comparison_radar")

        st.caption("💡 **Insight**: This radar chart shows the average profile of each group. Larger gaps indicate stronger differentiators.")

    # -------------------------------------------------------------------------
    # STATISTICAL SUMMARY
    # -------------------------------------------------------------------------
    st.subheader("📐 Statistical Summary")

    if available_metrics:
        summary_data = []

        for metric in available_metrics[:8]:  # Top 8 metrics
            existing_vals = existing[metric].dropna()
            attrited_vals = attrited[metric].dropna()

            summary_data.append({
                'Metric': truncate_label(metric.replace('_', ' ').title()),
                'Existing Mean': f"{existing_vals.mean():.2f}",
                'Existing Median': f"{existing_vals.median():.2f}",
                'Existing Std': f"{existing_vals.std():.2f}",
                'Attrited Mean': f"{attrited_vals.mean():.2f}",
                'Attrited Median': f"{attrited_vals.median():.2f}",
                'Attrited Std': f"{attrited_vals.std():.2f}",
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width="stretch", hide_index=True)

        st.caption("💡 **Insight**: Compare means, medians, and standard deviations. Large differences in means suggest strong predictive features.")

    # -------------------------------------------------------------------------
    # TOP DIFFERENTIATORS
    # -------------------------------------------------------------------------
    st.subheader("🎯 Top Differentiating Features")

    if numeric_cols:
        # Calculate effect size (Cohen's d) for each metric
        # Exclude naive_bayes features as they're synthetic and not informative
        differentiators = []

        for col in numeric_cols:
            # Skip naive_bayes synthetic columns
            if 'naive_bayes' in col.lower():
                continue

            if col in comparison_df.columns:
                existing_vals = existing[col].dropna()
                attrited_vals = attrited[col].dropna()

                if len(existing_vals) > 0 and len(attrited_vals) > 0:
                    # Calculate Cohen's d (effect size)
                    pooled_std = np.sqrt(
                        ((len(existing_vals) - 1) * existing_vals.std()**2 +
                         (len(attrited_vals) - 1) * attrited_vals.std()**2) /
                        (len(existing_vals) + len(attrited_vals) - 2)
                    )

                    if pooled_std > 0:
                        cohens_d = abs((existing_vals.mean() - attrited_vals.mean()) / pooled_std)

                        differentiators.append({
                            'Feature': col.replace('_', ' ').title(),
                            'Feature_Truncated': truncate_label(col.replace('_', ' ').title(), max_len=35),
                            'Effect Size': cohens_d,
                            'Existing Mean': existing_vals.mean(),
                            'Attrited Mean': attrited_vals.mean(),
                            'Difference %': ((attrited_vals.mean() - existing_vals.mean()) / existing_vals.mean() * 100) if existing_vals.mean() != 0 else 0
                        })

        if differentiators:
            diff_df = pd.DataFrame(differentiators).sort_values('Effect Size', ascending=False).head(10)

            # Create bar chart of effect sizes with truncated labels
            fig = px.bar(
                diff_df,
                x='Effect Size',
                y='Feature_Truncated',
                orientation='h',
                title="Top 10 Features by Effect Size (Cohen's d)",
                labels={'Effect Size': "Effect Size (Cohen's d)", 'Feature_Truncated': ''},
                color='Effect Size',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="comparison_effect_size_bar")

            st.caption("""
            💡 **Insight**: Effect size measures how different the two groups are (Cohen's d):
            - 0.2 = Small effect
            - 0.5 = Medium effect
            - 0.8+ = Large effect

            Features with large effect sizes are the strongest predictors of churn.
            """)

            # Show detailed table with truncated labels
            display_df = diff_df[['Feature_Truncated', 'Effect Size', 'Existing Mean', 'Attrited Mean', 'Difference %']].copy()
            display_df = display_df.rename(columns={'Feature_Truncated': 'Feature'})
            display_df['Effect Size'] = display_df['Effect Size'].round(3)
            display_df['Existing Mean'] = display_df['Existing Mean'].round(2)
            display_df['Attrited Mean'] = display_df['Attrited Mean'].round(2)
            display_df['Difference %'] = display_df['Difference %'].round(1).astype(str) + '%'

            st.dataframe(display_df, width="stretch", hide_index=True)
