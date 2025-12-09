# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/utils/msti_graphics_utils.py
# PURPOSE      : Visualization utilities and graphical parameters for MSTI projections
# DESCRIPTION  :
#   - Centralized visual configuration for all MSTI figures
#   - Vectorized repulsion forces for stabilizing 3D projections
#   - Automatic extraction of graphical parameters from MultiIndex
#   - Continent-based color palette and decade-based transparency
#   - Unified matplotlib style configuration with hierarchical font scaling
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import logging   # standardized logging
from typing import Dict, List, Optional, Tuple   # type hints

# 1.2 --- Third-party libraries ---
import numpy as np   # numerical operations
import pandas as pd   # dataframe manipulation
from scipy.spatial import cKDTree   # fast spatial indexing
import matplotlib.pyplot as plt   # static visualization
from scipy.interpolate import UnivariateSpline   # smooth curve fitting
from matplotlib.patches import Ellipse   # ellipse drawing for confidence regions
from scipy.stats import chi2   # chi-squared distribution for statistical tests

# 1.3 --- Internal modules ---
from src.config.msti_constants import (
    MSTI_INDEX_LABELS,
    REPULSION_DEFAULT_PARAMS
)

# ------------------------------------------------------------------------------------------------------------
# 2. Logger Initialization
# ------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)   # create module-level logger


# ------------------------------------------------------------------------------------------------------------
# 3. Constants
# ------------------------------------------------------------------------------------------------------------
CONVERGENCE_CHECK_INTERVAL = 5   # check convergence every N iterations
MIN_EPSILON = 1e-6   # minimum value to prevent division by zero
DEFAULT_ALPHA_RANGE = (0.3, 1.0)   # transparency range (oldest, newest)


# ------------------------------------------------------------------------------------------------------------
# 4. Continent and Country Color Palettes
# ------------------------------------------------------------------------------------------------------------
PALETTE_CONTINENT = {
    "Africa": "#1f77b4",
    "America": "#ff7f0e",
    "Asia": "#d62728",
    "Europe": "#2ca02c",
    "Oceania": "#8c564b",
}

PALETTE_COUNTRY = {
    "Allemagne": "#2ca02c",
    "Autriche": "#2ca02c",
    "Belgique": "#2ca02c",
    "Bulgarie": "#2ca02c",
    "Croatie": "#2ca02c",
    "Danemark": "#2ca02c",
    "Espagne": "#2ca02c",
    "Estonie": "#2ca02c",
    "Finlande": "#2ca02c",
    "France": "#2ca02c",
    "Grèce": "#2ca02c",
    "Hongrie": "#2ca02c",
    "Irlande": "#2ca02c",
    "Islande": "#2ca02c",
    "Italie": "#2ca02c",
    "Lettonie": "#2ca02c",
    "Lituanie": "#2ca02c",
    "Luxembourg": "#2ca02c",
    "Norvège": "#2ca02c",
    "Pays-Bas": "#2ca02c",
    "Pologne": "#2ca02c",
    "Portugal": "#2ca02c",
    "République slovaque": "#2ca02c",
    "Roumanie": "#2ca02c",
    "Royaume-Uni": "#2ca02c",
    "Slovénie": "#2ca02c",
    "Suède": "#2ca02c",
    "Suisse": "#2ca02c",
    "Tchéquie": "#2ca02c",
    "Türkiye": "#2ca02c",
    "Canada": "#ff7f0e",
    "Chili": "#ff7f0e",
    "Colombie": "#ff7f0e",
    "Costa Rica": "#ff7f0e",
    "États-Unis": "#ff7f0e",
    "Mexique": "#ff7f0e",
    "Argentine": "#ff7f0e",
    "Chine (République populaire de)": "#d62728",
    "Corée": "#d62728",
    "Israël": "#d62728",
    "Russie": "#d62728",
    "Japon": "#d62728",
    "Singapour": "#d62728",
    "Taipei chinois": "#d62728",
    "Afrique du Sud": "#1f77b4",
    "Australie": "#8c564b",
    "Nouvelle-Zélande": "#8c564b",
}


# ------------------------------------------------------------------------------------------------------------
# 5. Centralized MSTI Plot Style Configuration
# ------------------------------------------------------------------------------------------------------------
MSTI_PLOT_STYLE = {
    # 5.1 --- Figure dimensions and resolution ---
    "figure.figsize": (16, 8),   # unified aspect ratio (2:1)
    "figure.dpi": 150,   # display resolution
    "savefig.dpi": 300,   # export resolution
    "figure.autolayout": False,   # manual layout control

    # 5.2 --- Font hierarchy ---
    "font.size": 10,   # base font size
    "font.family": "sans-serif",   # font family
    "font.sans-serif": ["DejaVu Sans", "Arial"],   # preferred fonts
    
    # 5.3 --- Title hierarchy ---
    "axes.titlesize": 14,   # subplot title
    "axes.titleweight": "bold",   # bold subplot title
    "axes.titlepad": 12,   # spacing between title and plot
    
    # 5.4 --- Axis labels ---
    "axes.labelsize": 12,   # axis label font size
    "axes.labelweight": "normal",   # axis label weight
    "axes.labelpad": 8,   # spacing between label and axis
    
    # 5.5 --- Tick parameters ---
    "xtick.labelsize": 10,   # x-axis tick labels
    "ytick.labelsize": 10,   # y-axis tick labels
    "xtick.major.size": 4,   # major tick length (x)
    "ytick.major.size": 4,   # major tick length (y)
    "xtick.major.width": 0.8,   # major tick width (x)
    "ytick.major.width": 0.8,   # major tick width (y)
    "xtick.color": "#333333",   # tick color (x)
    "ytick.color": "#333333",   # tick color (y)
    
    # 5.6 --- Grid configuration ---
    "axes.grid": True,   # enable grid by default
    "axes.grid.axis": "both",   # grid on both axes
    "grid.linewidth": 0.6,   # grid line thickness
    "grid.alpha": 0.4,   # grid transparency
    "grid.linestyle": "--",   # dashed grid lines
    "grid.color": "#CCCCCC",   # light gray grid
    
    # 5.7 --- Spines (frame) configuration ---
    "axes.spines.top": False,   # hide top spine
    "axes.spines.right": False,   # hide right spine
    "axes.spines.left": True,   # show left spine
    "axes.spines.bottom": True,   # show bottom spine
    "axes.edgecolor": "#333333",   # spine color
    "axes.linewidth": 1.0,   # spine thickness
    
    # 5.8 --- Background and colors ---
    "axes.facecolor": "white",   # plot background
    "figure.facecolor": "white",   # figure background
    "savefig.facecolor": "white",   # saved figure background
    "axes.labelcolor": "#333333",   # label color
    
    # 5.9 --- Legend configuration ---
    "legend.frameon": False,   # no legend border
    "legend.fontsize": 9,   # legend font size (same as ticks)
    "legend.title_fontsize": 10,   # legend title font size
    "legend.borderpad": 0.4,   # padding inside legend
    "legend.labelspacing": 0.5,   # vertical spacing between entries
    "legend.handlelength": 1.5,   # marker line length
    "legend.handleheight": 0.7,   # marker height
    "legend.handletextpad": 0.5,   # spacing between marker and text
    "legend.columnspacing": 1.0,   # spacing between columns
    
    # 5.10 --- Line and marker defaults ---
    "lines.linewidth": 1.5,   # default line thickness
    "lines.markersize": 6,   # default marker size
    "scatter.marker": "o",   # default scatter marker
}


# ------------------------------------------------------------------------------------------------------------
# 6. Legend Marker Style Configuration
# ------------------------------------------------------------------------------------------------------------
LEGEND_MARKER_STYLE = {
    "marker": "o",   # circular markers
    "linestyle": "",   # no connecting lines
    "markersize": 8,   # marker size in legend
    "markeredgewidth": 0,   # no edge around markers
    "alpha": 0.9   # marker opacity
}


# ------------------------------------------------------------------------------------------------------------
# 7. Unified Matplotlib Style Application
# ------------------------------------------------------------------------------------------------------------
def standard_plot_style(
    fig: Optional[plt.Figure] = None,
    grid: bool = True,
    legend: bool = False,
    handles: Optional[List] = None,
    color_map: Optional[Dict] = None,
    legend_title: str = "Legend",
    main_title: Optional[str] = None,
    main_title_size: int = 16,
    tight_layout: bool = True
) -> None:
    """
    Apply unified STEA / MSTI visualization style with centralized configuration.

    This function configures matplotlib rcParams globally and optionally adds
    a standardized right-aligned legend with circular markers and hierarchical
    font scaling for consistent readability across all MSTI visualizations.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Target figure for legend and title (defaults to current figure).
    grid : bool, default=True
        Enable dashed grid background.
    legend : bool, default=False
        If True, add right-aligned external legend with circular markers.
    handles : list, optional
        Custom legend handles (if None, auto-generated from color_map).
    color_map : dict, optional
        Mapping of label → color for automatic legend generation.
    legend_title : str, default="Legend"
        Title displayed above legend entries.
    main_title : str, optional
        Centered suptitle for the entire figure (hierarchical typography).
    main_title_size : int, default=16
        Font size for main figure title.
    tight_layout : bool, default=True
        Apply tight_layout with legend spacing adjustments.
    
    Notes
    -----
    All visual parameters are centralized in MSTI_PLOT_STYLE to avoid
    redundancy across analysis modules. This ensures consistency in:
    - Figure dimensions and aspect ratios
    - Font hierarchy (titles, labels, ticks)
    - Grid styling (dashed lines)
    - Legend positioning and formatting
    """
    # 7.1 --- Apply global matplotlib parameters ---
    style = MSTI_PLOT_STYLE.copy()
    style["axes.grid"] = grid
    plt.rcParams.update(style)
    
    fig = fig or plt.gcf()
    
    # 7.2 --- Add main title ---
    if main_title:
        fig.suptitle(main_title, fontsize=main_title_size, fontweight="bold", y=0.98, ha="center")
    
    # 7.3 --- Add right-aligned legend ---
    if legend:
        if handles is None and color_map:
            handles = [plt.Line2D([], [], color=c, label=l, **LEGEND_MARKER_STYLE) 
                      for l, c in color_map.items()]
        
        if handles:
            fig.legend(
                handles=handles, title=legend_title, loc="center left",
                bbox_to_anchor=(0.98, 0.5), frameon=False,
                fontsize=MSTI_PLOT_STYLE["legend.fontsize"],
                title_fontsize=MSTI_PLOT_STYLE["legend.title_fontsize"],
                borderpad=MSTI_PLOT_STYLE["legend.borderpad"],
                labelspacing=MSTI_PLOT_STYLE["legend.labelspacing"],
                handlelength=MSTI_PLOT_STYLE["legend.handlelength"],
                handleheight=MSTI_PLOT_STYLE["legend.handleheight"],
                handletextpad=MSTI_PLOT_STYLE["legend.handletextpad"]
            )
    
    # 7.4 --- Apply tight layout ---
    if tight_layout:
        legend_space = 0.88 if legend else 1.0
        plt.tight_layout(rect=[0, 0, legend_space, 0.96 if main_title else 1.0])


# ------------------------------------------------------------------------------------------------------------
# 8. Extract Graphics Parameters from MultiIndex
# ------------------------------------------------------------------------------------------------------------
def extract_graphics_params(data: pd.DataFrame) -> Tuple[pd.Index, pd.Index, Dict[int, float]]:
    """
    Extract graphical parameters from MSTI MultiIndex for visualization.
    
    Extracts continent and decade from standard MultiIndex and computes
    transparency values scaled by decade (older = transparent, recent = opaque).
    
    Parameters
    ----------
    data : pandas.DataFrame
        Dataset with MSTI MultiIndex (zone, continent, year, decade).
    
    Returns
    -------
    tuple
        - continents (pd.Index): Continent labels per observation
        - decades (pd.Index): Decade labels per observation
        - alpha_dict (dict): Decade → transparency mapping
    """
    # 8.1 --- Extract continent and decade from MultiIndex ---
    continents = data.index.get_level_values(MSTI_INDEX_LABELS[1])   # continent level
    decades = data.index.get_level_values(MSTI_INDEX_LABELS[3])   # decade level
    
    # 8.2 --- Compute decade-based alpha (linear interpolation) ---
    unique_decades = sorted(decades.unique())   # sorted decade list
    n_decades = len(unique_decades)   # count decades
    alpha_min, alpha_max = DEFAULT_ALPHA_RANGE   # unpack transparency range

    # 8.3 --- Linear interpolation: oldest=0.3, newest=1.0 ---
    alpha_dict = {
        decade: alpha_min + (alpha_max - alpha_min) * (i / max(1, n_decades - 1))   # compute alpha
        for i, decade in enumerate(unique_decades)
    }   # dict comprehension to store transparency mapping
    
    return continents, decades, alpha_dict


# ------------------------------------------------------------------------------------------------------------
# 9. Vectorized Repulsion Forces
# ------------------------------------------------------------------------------------------------------------
def apply_repulsion_optimized(coords: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply optimized vectorized repulsion to reduce point overlap.
    
    Uses cKDTree for O(n log n) neighbor searches and NumPy broadcasting
    for vectorized force calculations. Prevents point clustering through
    inverse-distance repulsion forces with Gaussian kernel weighting.
    
    Algorithm:
    1. Build KD-Tree for efficient neighbor search
    2. For each point, find k nearest neighbors
    3. Compute repulsion forces: F = Î£(diff/dist) Ã— exp(-distÂ²/2ÏƒÂ²)
    4. Update positions with weighted forces
    5. Rebuild tree every N iterations for accuracy
    6. Check convergence periodically via mean displacement
    
    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates (n_observations Ã— 3).
    **kwargs : dict
        Override REPULSION_DEFAULT_PARAMS:
        - n_neighbors (int): Neighbors for repulsion (default: 20)
        - repulsion_strength (float): Gaussian kernel width (default: 0.3)
        - n_iterations (int): Max iterations (default: 50)
        - step_size (float): Update step (default: 0.05)
        - convergence_threshold (float): Early stopping (default: 1e-4)
        - rebuild_every (int): Rebuild KD-Tree frequency (default: 5)
    
    Returns
    -------
    numpy.ndarray
        Adjusted coordinates (n_observations Ã— 3).
    """
    # 9.1 --- Extract parameters ---
    params = {**REPULSION_DEFAULT_PARAMS, **kwargs}
    n_neighbors = min(params["n_neighbors"], max(1, coords.shape[0] - 1))
    kernel_width = params["repulsion_strength"]
    max_iter = params["n_iterations"]
    step_size = params["step_size"]
    threshold = params["convergence_threshold"]
    rebuild_every = params["rebuild_tree_every"]
    
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] < 2:
        return coords
    
    # 9.2 --- Precompute constants ---
    kernel_denominator = 2 * kernel_width ** 2
    
    def _query_neighbors(c):
        tree = cKDTree(c)
        return tree.query(c, k=n_neighbors + 1)

    distances, indices = _query_neighbors(coords)
    
    # 9.3 --- Iterative repulsion (SECTION COMPLÈTE À CONSERVER) ---
    for iteration in range(max_iter):
        # 9.3.1 --- Rebuild tree periodically ---
        if iteration and (iteration % rebuild_every == 0):
            distances, indices = _query_neighbors(coords)

        # 9.3.2 --- Get neighbor coords (exclude self at index 0) ---
        neighbor_coords = coords[indices[:, 1:]]
        
        # 9.3.3 --- Vector differences ---
        diff_vectors = coords[:, np.newaxis, :] - neighbor_coords
        
        # 9.3.4 --- Distances with safety margin ---
        dist_matrix = np.linalg.norm(diff_vectors, axis=2, keepdims=True) + MIN_EPSILON
        
        # 9.3.5 --- Gaussian kernel weights ---
        kernel_weights = np.exp(-dist_matrix**2 / kernel_denominator)
        
        # 9.3.6 --- Repulsion forces ---
        forces = np.sum((diff_vectors / dist_matrix) * kernel_weights, axis=1)
        
        # 9.3.7 --- Update positions ---
        displacement = step_size * forces
        coords += displacement
        
        # 9.3.8 --- Check convergence ---
        if iteration % CONVERGENCE_CHECK_INTERVAL == 0:
            mean_displacement = np.mean(np.linalg.norm(displacement, axis=1))
            if mean_displacement < threshold:
                logger.debug(f"Converged: iter={iteration+1}, disp={mean_displacement:.6f}")
                break
    
    return coords.astype(np.float32)


# ============================================================================================================
# 11. Spline-to-Spline Distance Metric
# ============================================================================================================
def compute_spline_barycenters(
    proj_df: pd.DataFrame,
    axes: Optional[list[str]] = None,
    time_col: str = "Decade",
    group_col: str = "Zone",
    spline_s: float = 2.0,
    n_points: int = 200,
    min_observations: int = 3
) -> pd.DataFrame:
    """
    Compute spline-based barycenters (continuous centers of gravity)
    for country trajectories in MFA space.

    Each country's barycenter is obtained by integrating the smoothed spline
    of its trajectory along the temporal axis. Includes robust protections
    against scale aberrations and numerical instabilities.

    Parameters
    ----------
    proj_df : pandas.DataFrame
        DataFrame containing MFA coordinates (F1, F2, F3, ...),
        a temporal column (e.g., 'Decade') and a grouping column (e.g., 'Zone').
    axes : list of str, optional
        Axes to use for barycenter computation (default: ['F1', 'F2', 'F3'] if available).
    time_col : str, default="Decade"
        Temporal variable to fit splines over.
    group_col : str, default="Zone"
        Column used to group trajectories (typically country name).
    spline_s : float, default=2.0
        Smoothing parameter for UnivariateSpline (higher = smoother, more stable).
    n_points : int, default=200
        Number of time grid points used for spline integration.
    min_observations : int, default=3
        Minimum number of temporal observations required (< 3 → skip).

    Returns
    -------
    pandas.DataFrame
        One row per country, columns = group_col + selected axes.
    
    Notes
    -----
    - Countries with < min_observations temporal points are excluded
    - Temporal normalization protected against division by zero
    - Post-integration clipping rejects values > 1000× median (scale protection)
    """
    # 11.1 --- Validate axes ---
    if axes is None:
        axes = [col for col in ["F1", "F2", "F3"] if col in proj_df.columns]
    if not axes:
        raise ValueError("No valid MFA axes found in input dataframe.")

    # 11.2 --- Compute barycenters per group ---
    results = []
    t_grid = np.linspace(0.0, 1.0, n_points)   # integration grid

    for country, group in proj_df.groupby(group_col):
        group = group.dropna(subset=axes + [time_col]).sort_values(time_col)
        
        if len(group) < min_observations:   # skip insufficient data
            continue

        # 11.2.1 --- Normalize temporal axis ---
        t = group[time_col].values.astype(float)
        t_range = t.max() - t.min()
        if t_range < 1e-6:   # identical decades
            continue
        t = (t - t.min()) / t_range   # normalize to [0,1]

        row_result = {group_col: country}

        # 11.2.2 --- Integrate spline per axis ---
        for ax in axes:
            y = group[ax].values
            try:
                spline = UnivariateSpline(t, y, s=spline_s)
                y_smooth = spline(t_grid)
                barycenter = np.trapz(y_smooth, t_grid)   # integrate over [0,1]
                
                if not np.isfinite(barycenter):   # reject NaN/Inf
                    raise ValueError(f"Non-finite barycenter: {barycenter}")
                row_result[ax] = barycenter
            except Exception:
                row_result[ax] = y.mean()   # fallback to mean

        results.append(row_result)

    # 11.3 --- Post-processing: clip extreme outliers ---
    if not results:
        return pd.DataFrame(columns=[group_col] + axes)
    
    df_result = pd.DataFrame(results).set_index(group_col)
    
    for ax in axes:
        median_val = df_result[ax].abs().median()
        threshold = max(1000 * median_val, 100)   # adaptive threshold
        df_result = df_result[df_result[ax].abs() <= threshold]   # clip outliers

    return df_result


# ============================================================================================================
# 12. Confidence Ellipse Drawing
# ============================================================================================================
def draw_confidence_ellipse(
    x, y, ax,
    confidence=0.95,
    edgecolor="black",
    linewidth=1.5,
    alpha=0.9,
):
    """
    Draw a confidence (or concentration) ellipse around (x, y) points.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the group points.
    ax : matplotlib.axes.Axes
        Target axes.
    n_std : float, default=2.0
        Confidence radius in standard deviations (≈95% if 2.447 for chi²).
    edgecolor : str, default="black"
        Border color.
    facecolor : str, default="none"
        Fill color.
    alpha : float, default=0.3
        Transparency of the fill.
    linewidth : float, default=1.0
        Line thickness.
    **kwargs : dict
        Additional Ellipse kwargs.
    """
    if len(x) < 2 or len(y) < 2:   # need at least 2 points for covariance
        return

    cov = np.cov(x, y)   # compute covariance matrix
    vals, vecs = np.linalg.eigh(cov)   # eigenvalues and eigenvectors
    order = vals.argsort()[::-1]   # sort indices in descending order
    vals, vecs = vals[order], vecs[:, order]   # reorder by importance

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))   # rotation angle in degrees
    width, height = 2 * np.sqrt(vals * chi2.ppf(confidence, df=2))   # ellipse axes from chi-squared

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),   # center at mean position
        width=width,   # major axis length
        height=height,   # minor axis length
        angle=theta,   # rotation angle
        edgecolor=edgecolor,   # border color
        facecolor="none",   # transparent fill
        lw=linewidth,   # line width
        alpha=alpha   # transparency level
    )
    ax.add_patch(ellipse)   # add ellipse to plot