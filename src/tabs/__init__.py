"""
Tab Modules - Modular Streamlit Tab Components

This package contains the tab rendering functions for the Credit Card
Customer Churn Analysis dashboard. Each tab is a separate module for
better code organization and maintainability.

============================================================================
WHAT IS THIS FILE? (__init__.py)
============================================================================

This special file makes the 'tabs' folder into a Python PACKAGE (not just a folder).

Key concepts:
1. Any folder with an __init__.py file becomes a Python package
2. This allows us to import from the folder: from src.tabs import ...
3. Without __init__.py, Python would treat 'tabs' as just a regular folder
4. The double underscores (dunder) mean this is a "magic" Python file with special meaning

WHY DO WE NEED THIS?
- It allows clean imports in app.py: from src.tabs import render_overview_tab
- Instead of: from src.tabs.overview import render_overview_tab (more verbose)
- It centralizes all our tab imports in one place
- It makes the code more maintainable and easier to refactor

============================================================================
"""

# ============================================================================
# IMPORTS - Bring in functions from our tab modules
# ============================================================================
# The dot (.) before each module name means "from THIS package" (relative import)
# This is called a RELATIVE IMPORT and it's the recommended way to import within a package

# Example: "from .overview" means "from the overview.py file in THIS folder"
# We then specify which function to import: "import render_overview_tab"

# Why relative imports (.overview) instead of absolute (src.tabs.overview)?
# - They work regardless of where the package is installed
# - They're more concise and easier to read
# - They make the package more portable and reusable

from .overview import render_overview_tab  # Tab 1: Dataset overview and KPIs
from .distributions import render_distributions_tab  # Tab 2: Feature distributions
from .churn_analysis import render_churn_analysis_tab  # Tab 3: Churn patterns
from .correlations import render_correlations_tab  # Tab 4: Feature correlations
from .customer_insights import render_customer_insights_tab  # Tab 5: Behavioral insights
from .churn_comparison import render_churn_comparison_tab  # Tab 6: Churn comparison (ignores filter)

# ============================================================================
# __all__ - Define what gets exported from this package
# ============================================================================
# __all__ is a list of strings defining the "public API" of this package
# When someone writes: from src.tabs import *
# Python will ONLY import the names listed in __all__

# Benefits of defining __all__:
# 1. Documentation: Shows users what functions they should use
# 2. Clean namespace: Prevents internal helper functions from being imported
# 3. IDE support: Helps code editors provide better autocomplete
# 4. Explicit is better than implicit: Clear about what's meant to be used externally

# In our case, we export all 6 render functions since they're all meant to be used by app.py
__all__ = [
    "render_overview_tab",
    "render_distributions_tab",
    "render_churn_analysis_tab",
    "render_correlations_tab",
    "render_customer_insights_tab",
    "render_churn_comparison_tab",
]

# ============================================================================
# HOW THIS WORKS IN PRACTICE
# ============================================================================
# In app.py, we can now write:
#   from src.tabs import render_overview_tab, render_distributions_tab, ...
#
# Instead of having to write:
#   from src.tabs.overview import render_overview_tab
#   from src.tabs.distributions import render_distributions_tab
#   from src.tabs.churn_analysis import render_churn_analysis_tab
#   from src.tabs.correlations import render_correlations_tab
#   from src.tabs.customer_insights import render_customer_insights_tab
#
# This __init__.py file acts as a "hub" that collects all the functions
# and makes them available in one convenient import statement!
