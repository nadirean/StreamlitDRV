"""
Metrics and analysis functions for dimensionality reduction evaluation.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_squared_error

from src.visualizations import (
    create_feature_error_plot,
    create_interactive_variance_plot,
    create_rfe_impact_plot,
    create_variance_plot,
)


def calculate_trustworthiness(X_original, X_embedded, n_neighbors=5):
    """Calculate trustworthiness of the embedding."""
    n_samples = X_original.shape[0]
    effective_neighbors = min(n_neighbors, n_samples - 1)
    if effective_neighbors < 1:
        return float("nan")
    return float(
        trustworthiness(X_original, X_embedded, n_neighbors=effective_neighbors)
    )


def analyze_pca_variance(X_scaled, feature_names, reducer):
    """Analyze PCA variance in detail."""
    st.subheader("Explained Variance Analysis")
    
    # Create a PCA with more components for analysis
    n_components_analysis = min(len(feature_names), X_scaled.shape[0] - 1)
    pca_full = PCA(n_components=n_components_analysis, random_state=42)
    pca_full.fit(X_scaled)
    
    # Explained variance ratio
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    # Individual explained variance plot
    fig_var = create_variance_plot(explained_var_ratio)
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Interactive cumulative variance plot with slider
    st.subheader("Interactive Cumulative Variance Explorer")
    
    # Slider for number of components
    max_components = len(cumulative_var_ratio)
    selected_components = st.slider(
        "Select number of components to analyze:",
        min_value=1,
        max_value=max_components,
        value=min(5, max_components),
        step=1,
        help="Move the slider to see cumulative variance for different numbers of components"
    )
    
    # Create interactive plot
    fig_interactive, selected_variance = create_interactive_variance_plot(
        cumulative_var_ratio, selected_components
    )
    st.plotly_chart(fig_interactive, use_container_width=True)
    
    # Display metrics for selected components
    display_variance_metrics(selected_components, max_components, selected_variance, cumulative_var_ratio)
    
    # Metrics table
    create_variance_metrics_table(cumulative_var_ratio)


def display_variance_metrics(selected_components, max_components, selected_variance, cumulative_var_ratio):
    """Display variance metrics in columns."""
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric(
            "Selected Components",
            selected_components,
            f"{selected_components/max_components:.1%} of total"
        )
    
    with col_metrics2:
        st.metric(
            "Cumulative Variance",
            f"{selected_variance:.3f}",
            f"{selected_variance:.1%}"
        )
    
    with col_metrics3:
        if selected_components < max_components:
            remaining_variance = cumulative_var_ratio[-1] - selected_variance
            st.metric(
                "Remaining Variance",
                f"{remaining_variance:.3f}",
                f"-{remaining_variance:.1%}"
            )
        else:
            st.metric(
                "Coverage",
                "Complete",
                "All variance captured"
            )


def create_variance_metrics_table(cumulative_var_ratio):
    """Create variance metrics table."""
    st.subheader("Variance Metrics")
    
    finite_mask = np.isfinite(cumulative_var_ratio)
    if not finite_mask.any():
        st.warning("Variance metrics are not available for this dataset.")
        return
    
    metrics_data = []
    
    # Find minimum components for different variance thresholds
    thresholds = [0.8, 0.9, 0.95, 0.99]
    for threshold in thresholds:
        exceeded = np.flatnonzero(finite_mask & (cumulative_var_ratio >= threshold))
        if exceeded.size > 0:
            min_components = int(exceeded[0]) + 1
            metrics_data.append({
                "Variance Threshold": f"{threshold*100}%",
                "Min Components Required": min_components,
                "Actual Variance Achieved": f"{cumulative_var_ratio[min_components - 1]:.3f}"
            })
    
    # Add current 2D projection metrics
    if cumulative_var_ratio.size >= 2 and np.isfinite(cumulative_var_ratio[1]):
        metrics_data.append({
            "Variance Threshold": "Current 2D",
            "Min Components Required": 2,
            "Actual Variance Achieved": f"{cumulative_var_ratio[1]:.3f}"
        })
    
    if not metrics_data:
        st.warning("No variance thresholds could be evaluated for this dataset.")
        return
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)


def analyze_reconstruction_error(X_scaled, reducer, feature_names, method):
    """Analyze reconstruction error for PCA."""
    st.subheader("Reconstruction Error")
    
    if method == "PCA":
        # Calculate reconstruction error
        X_reduced_2d = reducer.transform(X_scaled)
        X_reconstructed = reducer.inverse_transform(X_reduced_2d)
        reconstruction_error = mean_squared_error(X_scaled, X_reconstructed)
        
        st.metric("Mean Squared Error (MSE)", f"{reconstruction_error:.6f}")
        st.write("Lower MSE indicates better preservation of original data structure.")
        
        # Feature-wise reconstruction error
        feature_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=0)
        
        fig_feat_error = create_feature_error_plot(feature_names, feature_errors)
        st.plotly_chart(fig_feat_error, use_container_width=True)
    
    elif method == "KPCA":
        st.info("Reconstruction error calculation for Kernel PCA requires additional inverse mapping that may not be directly available.")


def analyze_feature_selection_impact(X_scaled, y, feature_names):
    """Analyze the impact of feature selection on variance preservation."""
    if len(feature_names) <= 3:  # Only if we have enough features
        return
    
    st.subheader("Feature Selection Impact")
    
    # RFE analysis
    st.write("Analyzing the impact of feature selection on variance preservation...")
    
    # Create different feature subsets using RFE
    feature_counts = [max(2, len(feature_names) // 4), max(3, len(feature_names) // 2), len(feature_names)]
    feature_counts = sorted(set(feature_counts))
    
    rfe_results = []
    
    for n_features in feature_counts:
        if n_features <= len(feature_names):
            # Use RFE with a simple estimator
            estimator = LinearRegression()
            rfe = RFE(estimator, n_features_to_select=n_features)
            X_rfe = rfe.fit_transform(X_scaled, y)
            
            # Apply PCA to the selected features
            pca_rfe = PCA(n_components=2, random_state=42)
            pca_rfe.fit(X_rfe)
            
            variance_2d = np.sum(pca_rfe.explained_variance_ratio_)
            selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
            
            rfe_results.append({
                "Number of Features": n_features,
                "Selected Features": ", ".join(selected_features[:3] + ["..."] if len(selected_features) > 3 else selected_features),
                "2D Variance Explained": f"{variance_2d:.3f}",
                "Variance Value": variance_2d
            })
    
    # Display results
    rfe_df = pd.DataFrame(rfe_results)
    st.dataframe(rfe_df[["Number of Features", "Selected Features", "2D Variance Explained"]], use_container_width=True)
    
    # Plot feature selection impact
    fig_rfe = create_rfe_impact_plot(rfe_results)
    st.plotly_chart(fig_rfe, use_container_width=True)


def show_metrics_analysis(X_scaled, y, feature_names, reducer, method):
    """Show comprehensive metrics and analysis."""
    st.header("5. Dimensionality Reduction Metrics")
    
    # Calculate metrics for methods that support them
    if method in ["PCA", "KPCA"]:
        # For PCA, we can calculate detailed variance metrics
        if method == "PCA":
            analyze_pca_variance(X_scaled, feature_names, reducer)
        
        # Reconstruction Error for PCA and KPCA
        analyze_reconstruction_error(X_scaled, reducer, feature_names, method)
        
        # Feature Selection Impact Analysis
        analyze_feature_selection_impact(X_scaled, y, feature_names)
    
    else:
        st.subheader("Limited Metrics")
        st.info(f"{method} is a non-linear method. Detailed variance analysis is not directly applicable, but the visualization above shows the method's ability to preserve local/global structure.")
