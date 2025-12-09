# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/visualization/msti_umap_projection.py
# PURPOSE      : 3D topological projection of MSTI observations using UMAP
# DESCRIPTION  :
#   - Generate 3D UMAP projections of imputed observations.
#   - Integrate HDBSCAN clustering on UMAP 3D embedding.
#   - Produce static (Matplotlib) and interactive (Plotly) visualizations.
#   - Apply vectorized repulsion for improved spatial distribution.
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import os   # operating system interfaces
from typing import Dict, Tuple   # type hints

# 1.2 --- Third-party libraries ---
import hdbscan   # density-based clustering
import matplotlib.pyplot as plt   # static visualization
import numba   # JIT compilation
import numpy as np   # numerical operations
import pandas as pd   # dataframe manipulation
import plotly.graph_objects as go   # interactive 3D visualization
import umap   # nonlinear dimensionality reduction
from matplotlib.cm import get_cmap   # colormap access
from matplotlib.colors import to_rgba   # color transparency management


# 1.3 --- Internal modules ---
from src.config.msti_graphics_utils import (
    apply_repulsion_optimized,
    extract_graphics_params,
    standard_plot_style,
    PALETTE_CONTINENT
)   # repulsion algorithm, visual parameters, plotting style, color palette
from src.config.msti_paths_config import OUTPUT_PATHS, DATA_PATHS   # standardized paths
from src.config.msti_constants import (
    DEFAULT_RANDOM_STATE,
    UMAP_DEFAULT_PARAMS,
    REPULSION_DEFAULT_PARAMS,
    get_numeric_columns
)   # default hyperparameters


# ------------------------------------------------------------------------------------------------------------
# 2. UMAP 3D Projection - Observations Visualization
# ------------------------------------------------------------------------------------------------------------
def project_umap3d_observations(
    imputed: pd.DataFrame,
    jitter_amplitude: float = 0.001,
    random_state: int = DEFAULT_RANDOM_STATE,
    apply_repulsion: bool = True,
    **override_params
) -> Dict:
    """
    Generate 3D UMAP projection of imputed MSTI observations.

    Nonlinear dimensionality reduction used to explore the topological structure
    of the dataset including proximity patterns, density variations, and natural
    groupings across continents and decades.
    
    [... reste du docstring ...]

    Returns
    -------
    dict
        - figure_plotly (plotly.graph_objects.Figure): Interactive 3D visualization
        - figure_mpl (matplotlib.figure.Figure): Static 3D visualization
        - metadata (dict): Number of observations, variables, and parameters used
    """
    print("\n=== STEP: UMAP 3D Projection (Observations) ===")

    # 2.1 --- Merge default parameters with overrides ---
    umap_params = {
        **UMAP_DEFAULT_PARAMS,
        **{k: v for k, v in override_params.items() if k in UMAP_DEFAULT_PARAMS}
    }   # merge UMAP-specific params
    repulsion_params = {
        **REPULSION_DEFAULT_PARAMS,
        **{k: v for k, v in override_params.items() if k in REPULSION_DEFAULT_PARAMS}
    }   # merge repulsion-specific params

    # --- Multithreading setup (Numba & OpenMP) ---
    N_THREADS = os.cpu_count()  # utilise tous les cœurs disponibles
    numba.set_num_threads(N_THREADS)
    os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
    print(f"Multithreading activé sur {numba.get_num_threads()} threads")

    np.random.seed(random_state)   # ensure deterministic behavior
    umap_params.pop("random_state", None)
    print(f"Numba threads: {numba.get_num_threads()}")

    # 2.2 --- Extract numeric variables ---
    numeric = get_numeric_columns(imputed)   # use centralized function
    n_obs = numeric.shape[0]   # count observations
    n_vars = numeric.shape[1]   # count variables
    print(f"Processing {n_obs} observations × {n_vars} variables")

    # 2.3 --- Add Gaussian jitter ---
    data_array = numeric.to_numpy()   # convert to NumPy array
    data_jittered = data_array + np.random.normal(0, jitter_amplitude, size=data_array.shape)   # add jitter

    # 2.4 --- Extract visual parameters ---
    continents, decades, alpha_dict = extract_graphics_params(imputed)   # continent, decade, transparency
    color_map = PALETTE_CONTINENT   # load continent color palette

    # 2.5 --- UMAP projection ---
    print(f"Running UMAP (n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']})")
    reducer = umap.UMAP(**umap_params)   # initialize UMAP reducer
    coords_3d = reducer.fit_transform(data_jittered)   # compute 3D embedding

    # 2.6 --- Apply vectorized repulsion (optional) ---
    if apply_repulsion:
        print("Applying vectorized repulsion for spatial stabilization...")
        coords_3d = apply_repulsion_optimized(coords=coords_3d, **repulsion_params)   # stabilize distribution

    # 2.7 --- Apply standard MSTI plot style ---
    standard_plot_style()   # configure matplotlib rcParams

    fig_mpl = plt.figure(figsize=(14, 10))   # use default figsize from PLOT_STYLE
    ax = fig_mpl.add_subplot(111, projection="3d")   # add 3D axes

    for continent, color in color_map.items():   # iterate over continents
        mask = (continents == continent)   # filter observations by continent
        alphas = np.array([alpha_dict[dec] for dec in decades[mask]])   # map decade to transparency

        ax.scatter(
            coords_3d[mask, 0],
            coords_3d[mask, 1],
            coords_3d[mask, 2],
            c=[to_rgba(color, a) for a in alphas],   # apply color with alpha
            s=3,   # marker size
            edgecolors="none",   # no edge color
            label=continent   # legend label
        )

    # 2.8 --- Configure static plot ---
    ax.set_xlabel("UMAP1")   # x-axis label
    ax.set_ylabel("UMAP2")   # y-axis label
    ax.set_zlabel("UMAP3")   # z-axis label
    ax.set_xlim(-15, 25)   # x-axis range
    ax.set_ylim(-15, 25)   # y-axis range
    ax.set_zlim(-15, 25)   # z-axis range

    plt.tight_layout()   # adjust layout

    # Apply unified style with legend
    standard_plot_style(
        fig=fig_mpl,
        grid=True,
        legend=True,
        color_map=color_map,
        legend_title="Continents",
        main_title=f"UMAP 3D Projection – {n_obs} observations",
        tight_layout=True
    )   # unified visual configuration

    plt.tight_layout()   # adjust layout

    # 2.10 --- Save static figure ---
    fig_path_mpl = OUTPUT_PATHS["figures"] / "msti_umap3d_observations_static.png"   # define output path
    fig_mpl.savefig(fig_path_mpl, bbox_inches="tight")   # DPI from rcParams
    print(f"Static figure saved to: {fig_path_mpl}")

    plt.show()   # display plot

    # 2.11 --- Interactive visualization (Plotly) ---
    traces = []   # initialize trace list
    for continent, color in color_map.items():   # iterate over continents
        mask = (continents == continent)   # filter observations

        traces.append(go.Scatter3d(
            x=coords_3d[mask, 0],
            y=coords_3d[mask, 1],
            z=coords_3d[mask, 2],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.8),   # marker style
            name=continent,   # legend label
        ))

    fig_plotly = go.Figure(data=traces)   # build Plotly figure
    fig_plotly.update_layout(
        title=f"Interactive UMAP 3D Projection ({n_obs} points)",
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3"
        ),
        showlegend=True,
    )   # configure layout

    # 2.12 --- Save interactive figure ---
    fig_path_plotly = OUTPUT_PATHS["figures"] / "msti_umap3d_observations.html"   # define output path
    fig_plotly.write_html(fig_path_plotly)   # save HTML file
    print(f"Interactive figure saved to: {fig_path_plotly}")

    return fig_plotly, fig_mpl   # return both figures


# ------------------------------------------------------------------------------------------------------------
# 3. UMAP 3D Projection - Cluster Visualization (HDBSCAN)
# ------------------------------------------------------------------------------------------------------------
def project_umap3d_clusters(
    imputed: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    jitter_amplitude: float = 0.001,
    random_state: int = DEFAULT_RANDOM_STATE,
    apply_repulsion: bool = True,
    **override_params
) -> Dict:
    """
    Generate 3D UMAP projection with HDBSCAN clustering.

    Nonlinear dimensionality reduction combined with density-based clustering
    to identify natural groupings in the MSTI dataset based on multidimensional
    proximity patterns independent of geographical or temporal labels.

    Algorithm:
    1. Extract numeric variables and add Gaussian jitter
    2. Apply UMAP dimensionality reduction
    3. Optionally apply vectorized repulsion
    4. Run HDBSCAN clustering on 3D coordinates
    5. Compute intra-cluster transparency (core = opaque, periphery = transparent)
    6. Generate static and interactive visualizations with cluster labels

    Parameters
    ----------
    imputed : pandas.DataFrame
        Imputed and indexed dataset with MSTI MultiIndex.
    jitter_amplitude : float, default=0.001
        Jitter amplitude before UMAP reduction.
    min_cluster_size : int, default=15
        Minimum cluster size for HDBSCAN (smaller = more granular clusters).
    min_samples : int, default=5
        Minimum local density for HDBSCAN core points.
    random_state : int, default from DEFAULT_RANDOM_STATE
        Random seed for reproducibility.
    apply_repulsion : bool, default=True
        If True, apply vectorized repulsion for spatial stabilization.
    **override_params : dict, optional
        Override default UMAP and repulsion parameters.

    Returns
    -------
    dict
        - figure_plotly (plotly.graph_objects.Figure): Interactive 3D visualization
        - figure_mpl (matplotlib.figure.Figure): Static 3D visualization
        - metadata (dict): Number of observations, variables, and parameters used
    """
    print("\n=== STEP: UMAP 3D Projection (Clusters) ===")

    # 3.1 --- Merge default parameters with overrides ---
    umap_params = {
        **UMAP_DEFAULT_PARAMS,
        **{k: v for k, v in override_params.items() if k in UMAP_DEFAULT_PARAMS}
    }   # merge UMAP-specific params
    repulsion_params = {
        **REPULSION_DEFAULT_PARAMS,
        **{k: v for k, v in override_params.items() if k in REPULSION_DEFAULT_PARAMS}
    }   # merge repulsion-specific params

    # --- Multithreading setup (Numba & OpenMP) ---
    N_THREADS = os.cpu_count()
    numba.set_num_threads(N_THREADS)
    os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
    print(f" Multithreading activé sur {numba.get_num_threads()} threads")

    np.random.seed(random_state)   # ensure deterministic behavior
    umap_params.pop("random_state", None)
    print(f"Numba threads: {numba.get_num_threads()}")

    # 3.2 --- Extract numeric variables ---
    numeric = get_numeric_columns(imputed)   # use centralized function
    n_obs = numeric.shape[0]   # count observations
    n_vars = numeric.shape[1]   # count variables
    print(f"Processing {n_obs} observations × {n_vars} variables")

    # 3.3 --- Add Gaussian jitter ---
    data_array = numeric.to_numpy()   # convert to NumPy array
    data_jittered = data_array + np.random.normal(0, jitter_amplitude, size=data_array.shape)   # add jitter

    # 3.4 --- UMAP projection ---
    print(f"Running UMAP (n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']})")
    reducer = umap.UMAP(**umap_params)   # initialize UMAP reducer
    coords_3d = reducer.fit_transform(data_jittered)   # compute 3D embedding

    # 3.5 --- Apply vectorized repulsion (optional) ---
    if apply_repulsion:
        print("Applying vectorized repulsion for spatial stabilization...")
        coords_3d = apply_repulsion_optimized(coords=coords_3d, **repulsion_params)   # stabilize distribution
        coords_3d = coords_3d.copy()   # ensure array ownership

    # 3.6 --- Apply standard MSTI plot style ---
    standard_plot_style()   # configure matplotlib rcParams

    # 3.7 --- HDBSCAN clustering ---
    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)   # initialize clusterer
    labels = clusterer.fit_predict(coords_3d)   # assign cluster labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)   # count valid clusters (exclude noise)
    n_noise = np.sum(labels == -1)   # count noise points
    print(f"Identified {n_clusters} clusters ({n_noise} noise points)")

    # 3.8 --- Compute intra-cluster alpha transparency ---
    alphas = np.zeros(len(coords_3d))   # initialize transparency array

    for label in np.unique(labels):   # iterate over cluster labels
        mask = (labels == label)   # filter points by cluster

        if not np.any(mask):   # skip empty clusters
            continue

        # Compute cluster center
        if label != -1:   # valid cluster
            center = coords_3d[mask].mean(axis=0)   # cluster centroid
        else:   # noise points
            center = coords_3d.mean(axis=0)   # global centroid

        # Compute distances to center
        distances = np.linalg.norm(coords_3d[mask] - center, axis=1)   # Euclidean distances

        # Normalize distances
        max_dist = distances.max()   # maximum distance
        if max_dist > 0:   # avoid division by zero
            dist_normalized = distances / max_dist   # normalize to [0, 1]
        else:
            dist_normalized = distances   # all points at center

        # Invert for transparency (core = opaque, periphery = transparent)
        alphas[mask] = 1 - dist_normalized

    alphas = np.clip(alphas, 0.2, 1.0)   # clamp alpha values to visible range

    # 3.9 --- Static visualization (Matplotlib) ---
    cmap = plt.colormaps.get_cmap("tab20")   # load colormap for clusters
    fig_mpl = plt.figure(figsize=(14, 10))   # create figure
    ax = fig_mpl.add_subplot(111, projection="3d")   # add 3D axes

    for label in sorted(set(labels)):   # iterate over clusters
        mask = (labels == label)   # filter points by cluster
        color = cmap(label % 20 if label != -1 else 0)   # assign cluster color
        mean_alpha = float(alphas[mask].mean())   # mean cluster transparency

        ax.scatter(
            coords_3d[mask, 0],
            coords_3d[mask, 1],
            coords_3d[mask, 2],
            s=3,   # marker size
            color=color,   # cluster color
            alpha=mean_alpha,   # transparency
            label=f"Cluster {label}" if label != -1 else "Noise"   # legend label
        )

   # 3.10 --- Configure static plot ---
    ax.set_xlabel("UMAP1", fontsize=11)   # x-axis label
    ax.set_ylabel("UMAP2", fontsize=11)   # y-axis label
    ax.set_zlabel("UMAP3", fontsize=11)   # z-axis label
    ax.set_xlim(-15, 25)   # x-axis range
    ax.set_ylim(-15, 25)   # y-axis range
    ax.set_zlim(-15, 25)   # z-axis range
    
    # Build cluster color map for legend
    cluster_color_map = {
        f"Cluster {l}" if l != -1 else "Noise": cmap(l % 20 if l != -1 else 0)
        for l in sorted(set(labels))
    }   # map cluster labels to colors
    
    # Apply unified style with legend
    standard_plot_style(
        fig=fig_mpl,
        grid=True,
        legend=True,
        color_map=cluster_color_map,
        legend_title="Clusters",
        main_title=f"UMAP 3D + HDBSCAN – {n_clusters} clusters ({n_obs} points)",
        tight_layout=True
    )   # unified visual configuration
    
    # 3.11 --- Save static figure ---
    fig_path_mpl = OUTPUT_PATHS["figures"] / "msti_umap3d_clusters_static.png"   # define output path
    fig_mpl.savefig(fig_path_mpl, bbox_inches="tight")   # high-resolution export
    print(f"Static figure saved to: {fig_path_mpl}")

    plt.show()   # display plot

    # 3.12 --- Interactive visualization (Plotly) ---
    traces = []   # initialize trace list

    for label in sorted(set(labels)):   # iterate over clusters
        mask = (labels == label)   # filter points by cluster
        rgba = tuple((np.array(cmap(label % 20 if label != -1 else 0)[:3]) * 255).astype(int)) + (180,)   # RGBA color
        color = "rgba({},{},{},{})".format(*map(int, rgba))   # format color string
        mean_alpha = float(alphas[mask].mean())   # mean cluster transparency

        traces.append(go.Scatter3d(
            x=coords_3d[mask, 0],
            y=coords_3d[mask, 1],
            z=coords_3d[mask, 2],
            mode="markers",
            marker=dict(size=3, color=color, opacity=mean_alpha),   # marker style
            name=f"Cluster {label}" if label != -1 else "Noise",   # legend label
        ))

    fig_plotly = go.Figure(data=traces)   # build Plotly figure
    fig_plotly.update_layout(
        title=f"Interactive UMAP 3D + HDBSCAN – {n_clusters} clusters ({n_obs} points)",
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3"
        ),
        showlegend=True,
    )   # configure layout

    # 3.13 --- Save interactive figure ---
    fig_path_plotly = OUTPUT_PATHS["figures"] / "msti_umap3d_clusters.html"   # define output path
    fig_plotly.write_html(fig_path_plotly)   # save HTML file
    print(f"Interactive figure saved to: {fig_path_plotly}")

    return fig_plotly, fig_mpl   # return both figures


# ------------------------------------------------------------------------------------------------------------
# 4. Standalone Module Test
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "="*60)
    print("MSTI UMAP PROJECTION - STANDALONE TEST")
    print("="*60)

    # 4.1 --- Load imputed dataset ---
    imputed_path = DATA_PATHS["processed"] / "msti_imputed.csv"
    print(f"\n[1/3] Loading imputed dataset...")
    imputed = pd.read_csv(imputed_path, sep=";", index_col=[0, 1, 2, 3])
    print(f"    ✓ Loaded: {imputed.shape}")

    # 4.2 --- Run observations projection ---
    print("\n[2/3] Running UMAP 3D projection (observations)...")
    fig_obs_plotly, fig_obs_mpl = project_umap3d_observations(imputed, apply_repulsion=True)
    print("    ✓ Observations projection complete")

    # 4.3 --- Run cluster projection ---
    print("\n[3/3] Running UMAP 3D projection (clusters)...")
    fig_clusters_plotly, fig_clusters_mpl = project_umap3d_clusters(
        imputed,
        min_cluster_size=15,
        min_samples=5,
        apply_repulsion=True
    )
    print("    ✓ Cluster projection complete")

    print("\n" + "="*60)
    print("✓ STANDALONE TEST COMPLETE")
    print("="*60)