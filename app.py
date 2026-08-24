"""
Main application file for StreamlitDRV.
"""
import streamlit as st

from src.config import PARAMETER_GUIDELINES, configure_page, show_header
from src.data_handler import (
    handle_missing_values,
    handle_sample_size,
    load_sample_dataset,
    load_user_dataset,
    scale_data,
    show_dataset_preview,
)
from src.metrics import show_metrics_analysis
from src.parameter_optimization import show_parameter_optimization
from src.reduction_methods import (
    apply_dimensionality_reduction,
    get_available_methods,
    show_performance_warning,
)
from src.visualizations import show_visualization_results


def main():
    """Main application function."""
    # Configure page and show header
    configure_page()
    show_header()
    
    # --- 1. Data Source Selection ---
    st.header("1. Select Data Source")
    
    data_source = st.radio(
        "Choose your data source:",
        ["Sample Dataset (Diabetes)", "Upload Your Own Data"]
    )
    
    # Load data based on selection
    if data_source == "Sample Dataset (Diabetes)":
        st.subheader("Loading Diabetes Dataset")
        X, y, feature_names, target_column, dataset_name = load_sample_dataset()
    else:
        X, y, feature_names, target_column, dataset_name = load_user_dataset()
    
    # Show dataset preview
    show_dataset_preview(X, y, feature_names, target_column, dataset_name, data_source)
    
    # --- 2. Preprocess Data ---
    st.header("2. Data Preprocessing")
    
    # Handle missing values
    X, y = handle_missing_values(X, y, feature_names)
    
    # Scale data
    X_scaled, _ = scale_data(X)
    
    # Handle sample size
    X_scaled, y = handle_sample_size(X_scaled, y)
    
    # --- 3. Choose Method ---
    st.header("3. Dimensionality Reduction")
    
    available_methods = get_available_methods()
    method = st.selectbox(
        "Choose dimensionality reduction method:", 
        available_methods
    )
    
    if "TRIMAP" not in available_methods:
        st.info(
            "TRIMAP is disabled because its optional dependencies are not installed. "
            "To enable it, run: uv sync --extra trimap (requires a C++ build toolchain)."
        )
    
    # Show performance warning for slow methods on large datasets
    show_performance_warning(method, X_scaled.shape[0])
    
    # Method-specific parameters
    st.sidebar.header("Parameters")
    
    # --- 4. Apply Reduction ---
    X_reduced, reducer = apply_dimensionality_reduction(X_scaled, method)
    
    # --- 5. Visualize Results ---
    show_visualization_results(X_reduced, y, method, target_column, dataset_name)
    
    # --- 6. Metrics and Analysis ---
    show_metrics_analysis(X_scaled, y, feature_names, reducer, method)
    
    # --- 7. Parameter Optimization ---
    show_parameter_optimization(X_scaled, method)
    
    # --- 8. Guidelines ---
    st.subheader("Parameter Selection Guidelines")
    
    if method in PARAMETER_GUIDELINES:
        st.info(PARAMETER_GUIDELINES[method])
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("StreamlitDRV")


if __name__ == "__main__":
    main()
