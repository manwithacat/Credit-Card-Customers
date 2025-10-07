"""
Credit Card Customer Churn Analysis Dashboard

A simple Streamlit app that visualizes credit card customer data
and basic churn patterns from the interim parquet snapshot.
"""

import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    """Load the interim parquet file with fallback to raw CSV."""
    root = pathlib.Path(__file__).resolve().parent
    interim = root / "data" / "interim" / "pre_transform_snapshot.parquet"

    if interim.exists():
        return pd.read_parquet(interim)

    # Fallback to raw CSV
    raw = root / "data" / "raw"
    try:
        csv_file = next(raw.rglob("*.csv"))
        return pd.read_csv(csv_file)
    except StopIteration:
        return pd.DataFrame()


# Page config
st.set_page_config(
    page_title="Credit Card Churn Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
df = load_data()

if df.empty:
    st.error("⚠️ No data found. Please run the ingestion notebook first.")
    st.stop()

# Header
st.title("💳 Credit Card Customer Churn Analysis")
st.markdown("""
Interactive dashboard for exploring credit card customer data and churn patterns.
Data loaded from interim parquet snapshot.
""")

# Sidebar filters
st.sidebar.header("Filters")

# Detect churn column
churn_col = next(
    (c for c in df.columns if c.lower() in {"attrition_flag", "churn", "churned"}),
    None
)

# Detect categorical columns for filtering
gender_col = next((c for c in df.columns if c.lower() in {"gender"}), None)
education_col = next(
    (c for c in df.columns if c.lower() in {"education_level", "education"}), None
)
card_col = next((c for c in df.columns if c.lower() in {"card_category"}), None)

# Apply filters
filtered_df = df.copy()

if churn_col and churn_col in df.columns:
    churn_filter = st.sidebar.multiselect(
        "Churn Status",
        options=df[churn_col].unique(),
        default=df[churn_col].unique()
    )
    filtered_df = filtered_df[filtered_df[churn_col].isin(churn_filter)]

if gender_col and gender_col in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=df[gender_col].dropna().unique(),
        default=df[gender_col].dropna().unique()
    )
    filtered_df = filtered_df[filtered_df[gender_col].isin(gender_filter)]

if card_col and card_col in df.columns:
    card_filter = st.sidebar.multiselect(
        "Card Category",
        options=df[card_col].dropna().unique(),
        default=df[card_col].dropna().unique()
    )
    filtered_df = filtered_df[filtered_df[card_col].isin(card_filter)]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distributions", "🔍 Churn Analysis", "🔗 Correlations"])

with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")

    with col2:
        st.metric("Total Features", len(filtered_df.columns))

    with col3:
        if churn_col and churn_col in filtered_df.columns:
            churn_count = (filtered_df[churn_col].str.contains("Attrited", case=False, na=False)).sum()
            st.metric("Churned Customers", f"{churn_count:,}")
        else:
            st.metric("Churned Customers", "N/A")

    with col4:
        if churn_col and churn_col in filtered_df.columns:
            churn_rate = (filtered_df[churn_col].str.contains("Attrited", case=False, na=False)).mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Churn Rate", "N/A")

    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        'Column': filtered_df.dtypes.index,
        'Type': filtered_df.dtypes.values.astype(str),
        'Non-Null': filtered_df.count().values,
        'Null %': ((1 - filtered_df.count() / len(filtered_df)) * 100).round(2).values
    })
    st.dataframe(dtype_df, use_container_width=True)

with tab2:
    st.header("Feature Distributions")

    # Numeric columns
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        col1, col2 = st.columns(2)

        with col1:
            selected_num = st.selectbox("Select Numeric Feature", numeric_cols, index=0)

        with col2:
            bins = st.slider("Number of Bins", 10, 100, 30)

        # Histogram
        fig = px.histogram(
            filtered_df,
            x=selected_num,
            nbins=bins,
            title=f"Distribution of {selected_num}",
            marginal="box",
            color=churn_col if churn_col else None
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

        with col2:
            # Box plots for selected features
            key_features = [c for c in numeric_cols if any(
                k in c.lower() for k in ['credit_limit', 'balance', 'transaction', 'utilization']
            )][:4]

            if key_features:
                fig = px.box(
                    filtered_df[key_features].melt(var_name='Feature', value_name='Value'),
                    x='Feature',
                    y='Value',
                    title="Key Features Box Plot"
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    # Categorical distributions
    st.subheader("Categorical Features")
    cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if cat_cols and len(cat_cols) > 0:
        selected_cat = st.selectbox("Select Categorical Feature", cat_cols)

        if selected_cat:
            value_counts = filtered_df[selected_cat].value_counts().head(15)

            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {selected_cat}",
                labels={'x': selected_cat, 'y': 'Count'}
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Churn Analysis")

    if churn_col and churn_col in filtered_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Churn distribution
            churn_counts = filtered_df[churn_col].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Churn Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn by card category
            if card_col and card_col in filtered_df.columns:
                churn_by_card = pd.crosstab(
                    filtered_df[card_col],
                    filtered_df[churn_col],
                    normalize='index'
                ) * 100

                fig = px.bar(
                    churn_by_card,
                    title=f"Churn Rate by {card_col}",
                    labels={'value': 'Percentage (%)', 'index': card_col},
                    barmode='group'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        # Churn by numeric features
        st.subheader("Churn by Numeric Features")

        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        key_numeric = [c for c in numeric_cols if any(
            k in c.lower() for k in ['credit_limit', 'balance', 'transaction', 'utilization', 'count', 'amount']
        )][:6]

        if key_numeric:
            col1, col2 = st.columns(2)

            for idx, feature in enumerate(key_numeric):
                with col1 if idx % 2 == 0 else col2:
                    fig = px.box(
                        filtered_df,
                        x=churn_col,
                        y=feature,
                        title=f"{feature} by Churn Status",
                        color=churn_col
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Churn by categorical features
        st.subheader("Churn by Categorical Features")

        cat_cols = [c for c in filtered_df.select_dtypes(include=['object', 'category']).columns
                    if c != churn_col]

        if cat_cols:
            selected_cat = st.selectbox("Select Feature for Churn Analysis", cat_cols, key='churn_cat')

            if selected_cat:
                churn_by_cat = pd.crosstab(
                    filtered_df[selected_cat],
                    filtered_df[churn_col],
                    normalize='index'
                ) * 100

                fig = px.bar(
                    churn_by_cat,
                    title=f"Churn Rate by {selected_cat}",
                    labels={'value': 'Percentage (%)', 'index': selected_cat},
                    barmode='group'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No churn column found in the dataset. Expected column name: 'Attrition_Flag'")

with tab4:
    st.header("Correlation Analysis")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        # Correlation heatmap
        corr_matrix = filtered_df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations
        st.subheader("Top Positive Correlations")

        # Get upper triangle of correlation matrix
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Strongest Positive Correlations**")
            st.dataframe(corr_df.head(10), use_container_width=True)

        with col2:
            st.write("**Strongest Negative Correlations**")
            st.dataframe(corr_df.tail(10), use_container_width=True)

        # Scatter plot for selected correlation
        st.subheader("Explore Feature Relationships")

        col1, col2 = st.columns(2)

        with col1:
            x_feature = st.selectbox("X-axis", numeric_cols, index=0, key='x_axis')

        with col2:
            y_feature = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key='y_axis')

        fig = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color=churn_col if churn_col else None,
            title=f"{x_feature} vs {y_feature}",
            trendline="ols" if len(filtered_df) > 10 else None,
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough numeric columns for correlation analysis.")

# Footer
st.markdown("---")
st.markdown("""
**Credit Card Customer Churn Analysis Dashboard**
Built with Streamlit • Data from Kaggle
""")
