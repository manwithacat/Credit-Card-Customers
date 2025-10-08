# Credit Card Customer Churn Analysis

A pedagogical data science project analyzing credit card customer attrition patterns using Python, pandas, and Streamlit for interactive visualization.

## Project Overview

This project demonstrates a streamlined data science workflow from raw data to interactive dashboard. Using real-world credit card customer data, we analyze churn patterns and present insights through an interactive Streamlit dashboard. The project emphasizes simplicity, code quality, and educational value.

**Dataset**: [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) from Kaggle
**Size**: 10,127 customers × 23 features
**Target Variable**: Attrition_Flag (Existing Customer vs Attrited Customer)

### Project Goals

✅ **Simple ETL Pipeline**
- Clean, well-documented data processing script
- Parquet file format for efficient storage
- Single command execution

✅ **Exploratory Data Analysis**
- Interactive visualization of distributions and relationships
- Dynamic filtering and segmentation
- Statistical summaries and correlation analysis

✅ **Interactive Dashboard**
- Multi-tab Streamlit application
- Real-time filtering across 10k+ records
- Professional visualizations with Plotly

✅ **Code Quality**
- Beginner-friendly extensive comments
- Linting (Ruff) and formatting (Black)
- Clean, maintainable codebase

## Project Structure

```
├── .github/
│   └── workflows/          # CI/CD pipelines (future)
├── data/
│   ├── raw/               # Raw CSV data (committed to repo)
│   └── processed/         # Pipeline outputs (parquet files)
│       ├── cleaned.parquet      # ETL output (27 columns)
│       └── features.parquet     # Feature engineering output (44 columns)
├── src/
│   ├── etl.py             # ETL pipeline script
│   └── features.py        # Feature engineering pipeline
├── app.py                 # Streamlit dashboard application
├── requirements.txt       # Python dependencies
├── Makefile              # Development automation
└── README.md             # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.12+ (or compatible version)
- Git for version control

### Setup & Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Credit-Card-Customers
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install
   # or: pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   make app
   # or: streamlit run app.py
   ```

   The dashboard will automatically run the ETL pipeline if needed! 🎉

### Optional: Run Pipeline Manually

If you want to regenerate the processed data:
```bash
make pipeline  # Runs ETL + feature engineering
# or step-by-step:
make etl       # python src/etl.py
make features  # python src/features.py
```

## Data Pipeline

Our three-stage pipeline transforms raw data into actionable insights:

```
Raw CSV → ETL → Feature Engineering → Dashboard
   ↓        ↓           ↓                 ↓
data/raw  etl.py   features.py         app.py
   ↓        ↓           ↓                 ↓
  1.5MB   cleaned   features          visualizations
        (27 cols)  (44 cols)
```

### Stage 1: ETL Process (`src/etl.py`)

The ETL script performs basic data cleaning and transformation:

1. **Extract**: Load raw CSV from `data/raw/BankChurners.csv`
2. **Transform**:
   - Standardize column names (lowercase, underscores)
   - Enforce data types (integers, floats, strings)
   - Create basic derived features (churn_flag, avg_transaction, tenure_years)
   - Handle missing values (replace "Unknown" with NA)
   - Clean string columns (strip whitespace)
   - Remove duplicate records
3. **Load**: Save as parquet to `data/processed/cleaned.parquet`

**Output**: 10,127 rows × 27 columns (487KB)

**Key Features**:
- Extensive pedagogical comments explaining each step
- Proper type handling (nullable Int64, float64, string dtypes)
- Basic feature creation (averages, tenure conversion)
- Robust error handling with try-except blocks

### Stage 2: Feature Engineering (`src/features.py`)

The feature engineering script creates 17 advanced predictive features:

**RFM (Recency, Frequency, Monetary) Metrics**
- `frequency_score`: Normalized transaction count (0-1 scale)
- `monetary_score`: Normalized transaction amount (0-1 scale)
- `rfm_score`: Combined RFM value score

**Customer Engagement Indicators**
- `engagement_score`: Composite of products held, activity, and inactivity

**Credit Health Metrics**
- `credit_to_limit_ratio`: Balance divided by credit limit
- `available_credit_ratio`: Available credit as % of limit
- `utilization_category`: Binned utilization (Low/Medium/High/Very High)

**Transaction Behavior Features**
- `trans_per_product`: Transaction frequency per product held
- `monthly_spend_rate`: Total spending normalized by tenure
- `transaction_trend`: Q4 vs Q1 change (Decreasing/Stable/Increasing)
- `transaction_density`: Transactions per month of tenure

**Customer Lifecycle**
- `lifecycle_stage`: Tenure-based segments (New/Growing/Mature/Loyal)

**Churn Risk Modeling**
- `churn_risk_score`: Composite risk metric (0=low, 1=high risk)
- `risk_category`: Binned risk levels (Low/Medium/High)

**Customer Segmentation**
- `customer_segment`: Value-based segments (Champion/At Risk/Potential/Hibernating)

**Interaction Features**
- `balance_estimate`: Credit limit × utilization
- `spend_per_product_monthly`: Spending per product per month

**Output**: 10,127 rows × 44 columns (1.1MB)

**Why Feature Engineering Matters**:
- Creates more predictive features than raw data
- Normalizes metrics for fair comparison (new vs long-term customers)
- Combines multiple signals into composite scores
- Enables customer segmentation for targeted strategies
- Supports both exploratory analysis and ML modeling

### Stage 3: Dashboard (`app.py`)

Interactive Streamlit application with 5 analytical tabs:

1. **📊 Overview**: Dataset summary, KPIs, data preview
2. **📈 Distributions**: Histograms, box plots, summary statistics
3. **🔍 Churn Analysis**: Pie charts, cross-tabulations, segment comparisons
4. **🔗 Correlations**: Heatmaps, scatter plots, relationship exploration
5. **💡 Customer Insights**: Engineered feature visualizations and advanced analytics

**Tab 5: Customer Insights** showcases the engineered features:
- **Churn Risk Score Analysis**: Validate engineered risk predictions
- **Customer Value Segmentation**: Champion/At Risk/Potential/Hibernating
- **RFM Analysis**: Frequency and monetary score distributions
- **Engagement Score**: Multi-factor engagement metric visualization
- **Lifecycle Analysis**: Churn rates by customer tenure stage
- **Advanced Metrics**: Transaction density, monthly spend rates

**Features**:
- Automatic pipeline execution (ETL → Features) if data missing
- Real-time filtering across all tabs (churn status, gender, card category)
- 21+ interactive Plotly visualizations
- Responsive layout with tabs and columns
- Insight captions explaining each visualization

## Development Workflow

### Make Commands

```bash
make help         # Show all available commands
make install      # Install dependencies
make lint         # Run Ruff linting
make format       # Run Black code formatting
make pre-commit   # Run lint + format (recommended before commits)
make etl          # Run ETL pipeline (raw → cleaned.parquet)
make features     # Run feature engineering (cleaned → features.parquet)
make pipeline     # Run full pipeline (ETL + features)
make app          # Run Streamlit dashboard
make clean        # Remove caches and temp files
```

### Code Quality Tools

**Ruff** - Fast Python linter
- Checks code quality and style
- Command: `make lint` or `ruff check .`

**Black** - Code formatter
- Ensures consistent code style
- Command: `make format` or `black .`

**Pre-commit Workflow**:
```bash
make pre-commit  # Run before committing changes
git add .
git commit -m "your message"
```

## Technology Stack

### Core Libraries

**pandas (2.3.3)** - Data manipulation
- DataFrame operations, cleaning, transformation
- Statistical summaries and aggregation

**numpy (2.0.2)** - Numerical computing
- Array operations and type handling

**pyarrow (21.0.0)** - Parquet support
- Efficient data storage (50-80% smaller than CSV)
- Preserves data types

### Visualization

**Streamlit (1.50.0)** - Dashboard framework
- Web application with zero frontend code
- `@st.cache_data` for performance optimization
- Interactive widgets (multiselect, slider, tabs)

**Plotly (6.3.1)** - Interactive charts
- Histograms, box plots, scatter plots, heatmaps
- Hover tooltips, zoom, and pan functionality

**Matplotlib (3.10.6) & Seaborn (0.13.2)** - Statistical plotting
- Additional visualization options

### Development Tools

**Ruff (0.8.4)** - Python linter (10-100x faster than alternatives)
**Black (24.10.0)** - Code formatter
**scikit-learn (1.6.1)** - Machine learning (future modeling)
**statsmodels (0.14.5)** - Statistical analysis (future modeling)

## Key Insights

### Dataset Overview

- **Total Records**: 10,127 customers
- **Raw Features**: 23 variables (16 numeric, 7 categorical)
- **Engineered Features**: 44 total variables (27 base + 17 engineered)
- **Churn Rate**: 16.1% (1,627 attrited customers)
- **Data Quality**: Zero missing values, no duplicates

### Data Pipeline Output

| Stage | Output File | Columns | Size | Description |
|-------|------------|---------|------|-------------|
| Raw | BankChurners.csv | 23 | 1.5MB | Original Kaggle dataset |
| ETL | cleaned.parquet | 27 | 487KB | Cleaned + basic features |
| Features | features.parquet | 44 | 1.1MB | Engineered features added |

### Churn Patterns

Based on dashboard exploration:

1. **Transaction Behavior**: Churned customers show lower transaction counts and amounts
2. **Card Distribution**: Blue Card holders constitute ~93% of customers
3. **Risk Segmentation**: High-risk customers (engineered score) have 2-3x higher churn rates
4. **Customer Segments**: "Hibernating" customers show highest churn, "Champions" lowest
5. **Lifecycle Trends**: New customers (0-1 year) have higher churn than loyal (3+ year) customers
6. **Engagement Impact**: Low engagement scores strongly correlate with churn

Explore these patterns interactively via the dashboard's 5 tabs and 21+ visualizations!

## Educational Focus

This project is designed for collaborative learning with the following pedagogical features:

### For Beginners

- **Extensive Comments**: Every section of code explained in detail
- **Concept Explanations**: Python idioms and pandas operations documented
- **Simple Structure**: Avoids over-engineering, focuses on fundamentals
- **Progressive Complexity**: Start simple (data loading) → build to advanced (interactive viz)

### Code Examples Covered

**Python Fundamentals:**
- List comprehensions and generator expressions
- Dictionary operations and comprehension
- Function definition and modularity (`if __name__ == "__main__"`)
- Error handling (try-except blocks)
- String operations and regex patterns
- Path handling with pathlib

**Data Science Techniques:**
- DataFrame manipulation (filtering, groupby, pivot, crosstab)
- Type handling (nullable types, type conversion)
- Feature engineering (normalization, binning, composite scores)
- Min-max scaling for feature normalization
- Categorical encoding (pd.cut for binning)
- Feature crosses (interaction features)
- RFM analysis implementation

### Best Practices Demonstrated

- Version control with Git
- Code formatting and linting
- Documentation and comments
- Modular code organization
- Defensive programming (error handling)
- Performance optimization (caching)

## Deployment

### Streamlit Cloud (Recommended)

1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Configure app settings
4. Deploy with auto-reload on commits

### Local Deployment

Simply run:
```bash
streamlit run app.py
```

The app automatically handles data processing on first run.

## Future Enhancements

Potential extensions for advanced learning:

- **Machine Learning**: Build churn prediction models (Random Forest, XGBoost)
- **Model Interpretability**: Add SHAP values for feature importance
- **Customer Segmentation**: K-means clustering analysis
- **Time Series**: Analyze churn trends over time
- **API Integration**: RESTful API for model predictions
- **Containerization**: Docker setup for reproducible deployment

## Contributing

This is a pedagogical group project. Contributions welcome via:

- Feature branches with descriptive names
- Pull requests with clear descriptions
- Code reviews with constructive feedback
- Documentation improvements

**Pre-commit Checklist**:
```bash
make pre-commit  # Lint and format
git add .
git commit -m "feat: descriptive message"
```

## Acknowledgments

- **Dataset**: [Sakshi Goyal](https://www.kaggle.com/sakshigoyal7) via Kaggle
- **Framework**: Built for collaborative learning and skill development

## License

This project is for educational purposes.

---

**Need Help?**
- Run `make help` to see all available commands
- Check inline code comments for explanations
- Review this README for workflow guidance
