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
    Load the interim parquet file with fallback to raw CSV.

    This function looks for our processed data file, and if it can't find it,
    falls back to loading the raw CSV file instead.

    Returns:
        pd.DataFrame: A pandas DataFrame containing our customer data
    """
    # __file__ is a special variable that contains the path to THIS script (app.py)
    # .resolve().parent gets the directory that contains this script
    root = pathlib.Path(__file__).resolve().parent

    # The / operator with Path objects joins paths together
    # This is like "root/data/interim/pre_transform_snapshot.parquet"
    interim = root / "data" / "interim" / "pre_transform_snapshot.parquet"

    # Check if the parquet file exists
    if interim.exists():
        # Parquet is a fast, compressed file format for data (better than CSV)
        return pd.read_parquet(interim)

    # Fallback to raw CSV if parquet doesn't exist
    raw = root / "data" / "raw"
    try:
        # rglob searches recursively for any .csv file in the raw directory
        # next() gets the first result from that search
        csv_file = next(raw.rglob("*.csv"))
        return pd.read_csv(csv_file)
    except StopIteration:
        # If no CSV is found, next() raises StopIteration error
        # Return an empty DataFrame instead of crashing
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
    st.error("⚠️ No data found. Please run the ingestion notebook first.")
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
Data loaded from interim parquet snapshot.
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
# This returns 4 tab objects that we assign to tab1, tab2, tab3, tab4
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distributions", "🔍 Churn Analysis", "🔗 Correlations"])

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
# FOOTER
# ============================================================================
st.markdown("---")  # Horizontal line
st.markdown("""
**Credit Card Customer Churn Analysis Dashboard**
Built with Streamlit • Data from Kaggle
""")
