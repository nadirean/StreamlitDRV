"""
Dimensionality reduction methods and their parameter configurations.
"""
import pacmap
import streamlit as st
import umap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

try:
    import trimap
    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False


def get_available_methods():
    """Return the dimensionality reduction methods available in this environment."""
    methods = ["PCA", "KPCA", "t-SNE", "UMAP"]
    if TRIMAP_AVAILABLE:
        methods.append("TRIMAP")
    methods.append("PaCMAP")
    return methods


def show_performance_warning(method, n_samples):
    """Show performance warnings for computationally intensive methods."""
    if n_samples > 5000 and method in ["t-SNE", "TRIMAP"]:
        st.warning(f"{method} can be slow on large datasets ({n_samples} samples). Consider using sampling or switching to UMAP/PCA for faster results.")
    elif n_samples > 10000 and method in ["KPCA", "PaCMAP"]:
        st.warning(f"{method} may take some time on datasets with {n_samples} samples. Consider using sampling for faster results.")


def create_pca_reducer(n_samples):
    """Create PCA reducer with default parameters."""
    return PCA(n_components=2, random_state=42)


def create_tsne_reducer(n_samples):
    """Create t-SNE reducer with configurable parameters."""
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
    perplexity = max(2, min(perplexity, n_samples - 1))
    return TSNE(n_components=2, perplexity=perplexity, random_state=42)


def create_umap_reducer(n_samples):
    """Create UMAP reducer with configurable parameters."""
    n_neighbors = st.sidebar.slider("Number of neighbors", 5, 100, 15)
    min_dist = st.sidebar.slider("Minimum distance", 0.0, 1.0, 0.1, 0.05)
    n_neighbors = max(2, min(n_neighbors, n_samples - 1))
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    )


def create_trimap_reducer(n_samples):
    """Create TRIMAP reducer with configurable parameters."""
    if not TRIMAP_AVAILABLE:
        raise ImportError(
            "TRIMAP requires the optional dependency group 'trimap'. "
            "Install it with: uv sync --extra trimap"
        )
    n_inliers = st.sidebar.slider("Number of inliers", 5, 50, 10)
    n_outliers = st.sidebar.slider("Number of outliers", 1, 20, 5)
    n_inliers = max(2, min(n_inliers, n_samples - 1))
    n_outliers = max(1, min(n_outliers, n_samples - 2))
    return trimap.TRIMAP(
        n_dims=2, n_inliers=n_inliers, n_outliers=n_outliers, verbose=False
    )


def create_kpca_reducer(n_samples):
    """Create Kernel PCA reducer with configurable parameters."""
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "poly", "sigmoid", "cosine"], index=0)
    gamma = st.sidebar.slider("Gamma (for rbf/poly/sigmoid)", 0.001, 10.0, 1.0)
    degree = st.sidebar.slider("Degree (for poly)", 2, 5, 3) if kernel == "poly" else 3
    return KernelPCA(
        n_components=2, kernel=kernel, gamma=gamma, degree=degree, random_state=42
    )


def create_pacmap_reducer(n_samples):
    """Create PaCMAP reducer with configurable parameters."""
    n_neighbors = st.sidebar.slider("Number of neighbors", 5, 100, 10)
    mn_ratio = st.sidebar.slider("MN ratio", 0.1, 1.0, 0.5)
    fp_ratio = st.sidebar.slider("FP ratio", 1.0, 4.0, 2.0)
    n_neighbors = max(2, min(n_neighbors, n_samples - 1))
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=n_neighbors,
        MN_ratio=mn_ratio, FP_ratio=fp_ratio, random_state=42
    )


def get_reducer(method, n_samples):
    """Get the appropriate reducer based on the selected method."""
    reducer_map = {
        "PCA": create_pca_reducer,
        "t-SNE": create_tsne_reducer,
        "UMAP": create_umap_reducer,
        "TRIMAP": create_trimap_reducer,
        "KPCA": create_kpca_reducer,
        "PaCMAP": create_pacmap_reducer
    }
    
    return reducer_map[method](n_samples)


def apply_dimensionality_reduction(X_scaled, method):
    """Apply dimensionality reduction to the scaled data."""
    reducer = get_reducer(method, X_scaled.shape[0])
    
    with st.spinner(f"Running {method}..."):
        X_reduced = reducer.fit_transform(X_scaled)
    
    return X_reduced, reducer
