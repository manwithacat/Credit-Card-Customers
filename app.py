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
                    return pd.read_csv(csv_file)
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
                return pd.read_parquet(cleaned_file)
            return pd.DataFrame()

    # Try to load the feature-engineered parquet file
    if features_file.exists():
        # Parquet is a fast, compressed file format for data (better than CSV)
        return pd.read_parquet(features_file)

    # Final fallback to cleaned data
    if cleaned_file.exists():
        st.warning("⚠️ Loading cleaned data without engineered features.")
        return pd.read_parquet(cleaned_file)

    # Last resort: raw CSV
    st.warning("⚠️ Loading raw CSV as fallback. Data may not be fully processed.")
    raw = root / "data" / "raw"
    try:
        csv_file = next(raw.rglob("*.csv"))
        return pd.read_csv(csv_file)
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
# This returns 5 tab objects that we assign to tab1, tab2, tab3, tab4, tab5
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Distributions",
    "🔍 Churn Analysis",
    "🔗 Correlations",
    "💡 Customer Insights"
])

# ============================================================================
# TAB 1: OVERVIEW - High-level metrics and data preview
# ============================================================================
# "with tab1:" means everything indented below will appear inside tab1
# This is Python's context manager syntax
with tab1:
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
        st.metric("Total Features", len(filtered_df.columns))

    with col3:
        if churn_col and churn_col in filtered_df.columns:
            # .str.contains() searches for text within string values
            # case=False means ignore capitalization
            # na=False means treat missing values as False
            # .sum() counts how many True values (rows with "Attrited")
            churn_count = (filtered_df[churn_col].str.contains("Attrited", case=False, na=False)).sum()
            st.metric("Churned Customers", f"{churn_count:,}")
        else:
            st.metric("Churned Customers", "N/A")

    with col4:
        if churn_col and churn_col in filtered_df.columns:
            # .mean() gives proportion of True values (0 to 1)
            # * 100 converts to percentage
            # :.1f formats as decimal with 1 place after decimal point
            churn_rate = (filtered_df[churn_col].str.contains("Attrited", case=False, na=False)).mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Churn Rate", "N/A")

    # -------------------------------------------------------------------------
    # DATA PREVIEW - Show first 100 rows
    # -------------------------------------------------------------------------
    st.subheader("Sample Data")
    # st.dataframe() creates an interactive table that users can scroll and sort
    # .head(100) gets the first 100 rows
    # use_container_width=True makes the table expand to fill available width
    st.dataframe(filtered_df.head(100), use_container_width=True)

    # -------------------------------------------------------------------------
    # DATA TYPES TABLE - Show column info and missing values
    # -------------------------------------------------------------------------
    st.subheader("Data Types")
    # Create a new DataFrame that summarizes information about our data
    # This is a dictionary where keys become column names
    dtype_df = pd.DataFrame({
        'Column': filtered_df.dtypes.index,  # Column names
        'Type': filtered_df.dtypes.values.astype(str),  # Data types (int, float, object, etc.)
        'Non-Null': filtered_df.count().values,  # Number of non-missing values
        'Null %': ((1 - filtered_df.count() / len(filtered_df)) * 100).round(2).values  # % missing
    })
    st.dataframe(dtype_df, use_container_width=True)

# ============================================================================
# TAB 2: DISTRIBUTIONS - Explore how values are spread across features
# ============================================================================
with tab2:
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
            color=churn_col if churn_col else None  # Color bars by churn status if available
        )
        # st.plotly_chart() displays the Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------------------
        # SUMMARY STATISTICS TABLE
        # -------------------------------------------------------------------------
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)

        with col1:
            # .describe() generates statistics: count, mean, std, min, quartiles, max
            st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

        with col2:
            # Box plots for key financial features
            # This is a LIST COMPREHENSION - a Python shortcut for filtering lists
            # It reads: "include column c IF any keyword k is in the column name"
            # [:4] takes only the first 4 matches
            key_features = [c for c in numeric_cols if any(
                k in c.lower() for k in ['credit_limit', 'balance', 'transaction', 'utilization']
            )][:4]

            if key_features:
                # .melt() transforms data from wide to long format
                # Wide: each feature is a column → Long: one "Feature" column, one "Value" column
                # This makes it easier to plot multiple features on one chart
                fig = px.box(
                    filtered_df[key_features].melt(var_name='Feature', value_name='Value'),
                    x='Feature',
                    y='Value',
                    title="Key Features Box Plot"
                )
                # Rotate x-axis labels 45 degrees so they don't overlap
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # CATEGORICAL FEATURES SECTION
    # -------------------------------------------------------------------------
    st.subheader("Categorical Features")
    # Select non-numeric columns (text categories like "Male/Female", "Gold/Silver", etc.)
    cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

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
                labels={'x': selected_cat, 'y': 'Count'}
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: CHURN ANALYSIS - Understand who churned and why
# ============================================================================
with tab3:
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
                names=churn_counts.index,    # The labels for each slice
                title="Churn Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # -------------------------------------------------------------------------
            # CHURN RATE BY CARD CATEGORY
            # -------------------------------------------------------------------------
            if card_col and card_col in filtered_df.columns:
                # pd.crosstab() creates a cross-tabulation (contingency table)
                # It counts how many rows fall into each combination of categories
                # normalize='index' converts counts to percentages (each row sums to 100%)
                churn_by_card = pd.crosstab(
                    filtered_df[card_col],
                    filtered_df[churn_col],
                    normalize='index'
                ) * 100  # Convert from 0-1 to 0-100

                fig = px.bar(
                    churn_by_card,
                    title=f"Churn Rate by {card_col}",
                    labels={'value': 'Percentage (%)', 'index': card_col},
                    barmode='group'  # Put bars side-by-side instead of stacked
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------------------
        # CHURN BY NUMERIC FEATURES - Compare distributions between churned/not churned
        # -------------------------------------------------------------------------
        st.subheader("Churn by Numeric Features")

        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        # List comprehension: filter to key financial columns, take first 6
        key_numeric = [c for c in numeric_cols if any(
            k in c.lower() for k in ['credit_limit', 'balance', 'transaction', 'utilization', 'count', 'amount']
        )][:6]

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
                        color=churn_col
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------------------
        # CHURN BY CATEGORICAL FEATURES
        # -------------------------------------------------------------------------
        st.subheader("Churn by Categorical Features")

        # Get all categorical columns except the churn column itself
        cat_cols = [c for c in filtered_df.select_dtypes(include=['object', 'category']).columns
                    if c != churn_col]

        if cat_cols:
            # key='churn_cat' gives this widget a unique ID so Streamlit can track it
            # This prevents conflicts if we have another selectbox with same label elsewhere
            selected_cat = st.selectbox("Select Feature for Churn Analysis", cat_cols, key='churn_cat')

            if selected_cat:
                # Same crosstab pattern as before - show churn % for each category
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
        # st.warning() shows a yellow warning message
        st.warning("No churn column found in the dataset. Expected column name: 'Attrition_Flag'")

# ============================================================================
# TAB 4: CORRELATIONS - Find relationships between numeric features
# ============================================================================
with tab4:
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

        # px.imshow() creates a heatmap (2D color-coded matrix)
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',  # Show correlation values on cells, formatted to 2 decimals
            aspect='auto',    # Automatically adjust aspect ratio
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r',  # Red-Blue colorscale (red=negative, blue=positive)
            zmin=-1,  # Minimum value for color scale
            zmax=1    # Maximum value for color scale
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------------------
        # TOP CORRELATIONS TABLE
        # -------------------------------------------------------------------------
        st.subheader("Top Positive Correlations")

        # Extract upper triangle of correlation matrix to avoid duplicates
        # A correlation matrix is symmetric (corr(A,B) = corr(B,A))
        # We only need to look at pairs once
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):  # j starts at i+1 (upper triangle)
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

        # Convert to DataFrame and sort by correlation strength
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Strongest Positive Correlations**")
            # .head(10) gets top 10 rows (strongest positive correlations)
            st.dataframe(corr_df.head(10), use_container_width=True)

        with col2:
            st.write("**Strongest Negative Correlations**")
            # .tail(10) gets bottom 10 rows (strongest negative correlations)
            st.dataframe(corr_df.tail(10), use_container_width=True)

        # -------------------------------------------------------------------------
        # SCATTER PLOT - Visualize relationship between two features
        # -------------------------------------------------------------------------
        st.subheader("Explore Feature Relationships")

        col1, col2 = st.columns(2)

        with col1:
            # Let user pick which features to compare
            x_feature = st.selectbox("X-axis", numeric_cols, index=0, key='x_axis')

        with col2:
            # Default to second column, or first if only one column exists
            # min() ensures we don't go out of bounds
            y_feature = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key='y_axis')

        # px.scatter() creates a scatter plot (each point is a customer)
        fig = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color=churn_col if churn_col else None,  # Color points by churn status
            title=f"{x_feature} vs {y_feature}",
            trendline="ols" if len(filtered_df) > 10 else None,  # Add linear regression line if enough data
            opacity=0.6  # Make points slightly transparent so we can see overlaps
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough numeric columns for correlation analysis.")

# ============================================================================
# TAB 5: CUSTOMER INSIGHTS - Advanced behavioral pattern analysis
# ============================================================================
with tab5:
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
            st.dataframe(summary.round(2), use_container_width=True)

            st.caption("💡 **Insight**: Compare average behaviors between customer segments to identify churn indicators.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")  # Horizontal line
st.markdown("""
**Credit Card Customer Churn Analysis Dashboard**
Built with Streamlit • Data from Kaggle
""")
