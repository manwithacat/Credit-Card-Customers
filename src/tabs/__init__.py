"""
Tab Modules - Modular Streamlit Tab Components

This package contains the tab rendering functions for the Credit Card
Customer Churn Analysis dashboard. Each tab is a separate module for
better code organization and maintainability.
"""

from .overview import render_overview_tab
from .distributions import render_distributions_tab
from .churn_analysis import render_churn_analysis_tab
from .correlations import render_correlations_tab
from .customer_insights import render_customer_insights_tab

__all__ = [
    "render_overview_tab",
    "render_distributions_tab",
    "render_churn_analysis_tab",
    "render_correlations_tab",
    "render_customer_insights_tab",
]
