# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/analysis/msti_cah_mfa.py
# PURPOSE      : Advanced multivariate analysis of imputed MSTI variables (CAH + MFA)
# DESCRIPTION  :
#   - Hierarchical clustering (CAH) for automatic grouping of homogeneous variables.
#   - Multiple factor analysis (MFA) for exploring multidimensional structures.
#   - Correlation circles for variable contributions visualization.
#   - Observation projections by continent and country with spline-smoothed trajectories.
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import warnings   # runtime warnings
from typing import Dict, Tuple   # type hints

warnings.filterwarnings('ignore')   # suppress non-critical warnings

# 1.2 --- Third-party libraries ---
import matplotlib.pyplot as plt   # static visualization
import numpy as np   # numerical operations
import pandas as pd   # dataframe manipulation
from adjustText import adjust_text   # automatic text label adjustment
from matplotlib.patheffects import withStroke   # text halo for readability
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA, SparsePCA   # principal component analysis (standard + sparse)
from sklearn.metrics import silhouette_score   # clustering evaluation

# 1.3 --- Internal modules ---
from src.config.msti_graphics_utils import (
    PALETTE_CONTINENT,
    PALETTE_COUNTRY,
    extract_graphics_params,
    standard_plot_style,
    compute_spline_barycenters,
    draw_confidence_ellipse
)   # visual utilities
from src.config.msti_variables_mapping import VARIABLE_BLOCKS, COUNTRY_TO_CONTINENT   # variable organization
from src.config.msti_paths_config import OUTPUT_PATHS, DATA_PATHS   # standardized paths
from src.config.msti_constants import get_numeric_columns, DEFAULT_RANDOM_STATE, MSTI_INDEX_LABELS   # numeric column extraction


# ------------------------------------------------------------------------------------------------------------
# 2. Hierarchical Clustering of Variables (CAH)
# ------------------------------------------------------------------------------------------------------------
def cluster_variables_hierarchical(imputed: pd.DataFrame) -> Dict:
    """
    Perform hierarchical clustering (CAH) on MSTI variables to identify 
    homogeneous groups.

    Uses Euclidean distance and Ward linkage criterion to minimize intra-class 
    variance. Optimal number of clusters determined via elbow method (inertia 
    acceleration) and silhouette score maximization.

    Parameters
    ----------
    imputed : pandas.DataFrame
        Dataset after imputation containing all MSTI indicator variables.

    Returns
    -------
    dict
        - data (None): No transformed data
        - metadata (dict): n_clusters_elbow, n_clusters_silhouette, best_silhouette
        - figures (dict): Paths to PNG files
        - exports (None): No exports
    """
    print("\n=== STEP: Hierarchical Clustering Analysis ===")
    
    # 2.1 --- Extract numeric variables ---
    numeric  = get_numeric_columns(imputed)   # use centralized function
    n_vars = len(numeric.columns)   # count variables
    print(f"Clustering {n_vars} variables")
    
    # 2.2 --- Prepare data for clustering ---
    vars_values = numeric.T.values   # transpose: variables as rows
    vars_names = numeric.columns.tolist()
    linkage_mat = linkage(vars_values, method="ward", metric="euclidean")   # compute linkage matrix
    
    # 2.3 --- Elbow method for optimal cluster count ---
    last_distances = linkage_mat[-20:, 2]   # last 20 fusion distances
    acceleration = np.diff(last_distances, 2)   # second derivative (acceleration)
    n_clusters_elbow = acceleration.argmax() + 2   # optimal k from elbow
    cut_threshold = linkage_mat[-n_clusters_elbow, 2]   # cut height
    
    # 2.4 --- Static visualization: Dendrogram ---
    fig_dendro, ax = plt.subplots(figsize=(16, 8))
    
    dendrogram(
        linkage_mat,
        labels=vars_names,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=cut_threshold,
        above_threshold_color="grey",
        ax=ax
    )   # generate dendrogram
    
    # 2.4.1 --- Configure dendrogram axes ---
    ax.set_ylabel("Fusion distance")   # y-axis label
    ax.set_xlabel("Variables")   # x-axis label
    ax.axhline(
        y=cut_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Cut threshold (k ≈ {n_clusters_elbow})"
    )   # add cut line

    ax.grid(True, axis='both', linestyle='--', linewidth=0.6, alpha=0.4, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # 2.4.2 --- Configure y-axis ---
    y_max = np.max(linkage_mat[:, 2]) * 1.05   # maximum y value with margin
    ax.set_ylim(0, y_max)   # set y limits
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    # 2.4.3 --- Apply unified style ---
    standard_plot_style(
        fig=fig_dendro,
        grid=True,
        legend=False,
        main_title="Variable Dendrogram (CAH – Ward, Euclidean distance)",
        main_title_size=16,
        tight_layout=True
    )
    
    plt.show()   # display plot
    
    # 2.4.4 --- Save dendrogram ---
    dendro_path = OUTPUT_PATHS["figures"] / "msti_cah_dendrogram.png"   # define output path
    fig_dendro.savefig(dendro_path, bbox_inches="tight", facecolor="white")   # DPI from rcParams
    
    # 2.5 --- Evaluation: inertia & silhouette ---
    max_clusters = 30   # maximum clusters to test
    inertia_list = []   # initialize inertia results
    silhouette_list = []   # initialize silhouette results
    best_silhouette = -1   # initialize best silhouette tracker
    n_clusters_silhouette = None   # initialize best k tracker
        
    for k in range(2, max_clusters + 1):   # loop over cluster counts
        labels_k = fcluster(linkage_mat, k, criterion='maxclust')   # partition into k clusters
        
        # 2.5.1 --- Compute intra-cluster inertia ---
        inertia_k = np.sum([
            np.sum((vars_values[labels_k == i] - np.mean(vars_values[labels_k == i], axis=0)) ** 2)
            for i in np.unique(labels_k)
        ])   # sum of squared distances to cluster centers
        inertia_list.append(inertia_k)   # store inertia
        
        # 2.5.2 --- Compute silhouette score ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            silhouette_k = silhouette_score(vars_values, labels_k)   # measure cluster quality
        silhouette_list.append(silhouette_k)   # store silhouette
        
        # 2.5.3 --- Update best if improved ---
        if silhouette_k > best_silhouette:
            best_silhouette = silhouette_k   # update best score
            n_clusters_silhouette = k   # update best k
    
    # 2.5.4 --- Convert inertia to percentage ---
    total_inertia = np.sum((vars_values - vars_values.mean(axis=0)) ** 2)   # compute total inertia
    inertia_pct = [100 * (1 - inertia / total_inertia) for inertia in inertia_list]   # convert to percentage
    k_values = range(2, max_clusters + 1)   # cluster range for plotting
    
    # 2.6 --- Static visualization: Evaluation plot ---
    fig_eval, ax1 = plt.subplots(figsize=(16, 8))
    
    # 2.6.1 --- Plot inertia on primary axis ---
    color_inertia = "#1f77b4"   # blue color for inertia
    ax1.plot(k_values, inertia_pct, "o-", color=color_inertia, lw=2, label="Explained inertia (%)")   # plot inertia
    ax1.set_xlabel("Number of clusters (k)")   # x-axis label
    ax1.set_ylabel("Explained inertia (%)", color=color_inertia)   # y-axis label
    ax1.tick_params(axis='y', labelcolor=color_inertia)   # set tick color
    ax1.set_ylim(0, 105)   # set y limits with margin
    
    # 2.6.2 --- Plot silhouette on secondary axis ---
    color_silhouette = "#8B0000"   # orange color for silhouette
    ax2 = ax1.twinx()   # create secondary axis
    ax2.grid(False)
    ax2.plot(k_values, silhouette_list, "s--", color=color_silhouette, lw=2, label="Silhouette score")   # plot silhouette
    ax2.set_ylabel("Silhouette score", color=color_silhouette)   # y-axis label
    ax2.tick_params(axis='y', labelcolor=color_silhouette)   # set tick color
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_ticks_position("right")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#333333")
    ax2.spines["right"].set_linewidth(1.0)
    
    # 2.6.3 --- Add optimal k marker ---
    k_optimal = n_clusters_silhouette   # optimal k from silhouette
    ax1.axvline(x=k_optimal, color="red", linestyle="--", lw=1.5)   # add vertical line
    ax1.text(k_optimal + 0.5, 5, f"k = {k_optimal}", color="red", va="bottom", ha="left")   # add label
    
    # 2.6.4 --- Configure legend ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()   # get primary axis legend
    lines_2, labels_2 = ax2.get_legend_handles_labels()   # get secondary axis legend
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),   # right-aligned with consistent spacing
        frameon=False   # no border
    )   # combined legend
    
    # 2.6.5 --- Apply unified style ---
    standard_plot_style(
        fig=fig_eval,
        grid=True,
        legend=False,   # legend already created
        main_title="CAH Comparative Analysis – Inertia vs Silhouette",
        main_title_size=16,
        tight_layout=False   # manual adjustment for twin axes
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])   # reserve space for legend and title
    plt.show()   # display plot
    
    # 2.6.6 --- Save evaluation plot ---
    eval_path = OUTPUT_PATHS["figures"] / "msti_cah_evaluation.png"   # define output path
    fig_eval.savefig(eval_path, bbox_inches="tight", facecolor="white")   # DPI from rcParams
    
    # 2.7 --- Summary ---
    print(f"\n=== Results ===")
    print(f"  • Elbow method: {n_clusters_elbow} clusters")
    print(f"  • Max silhouette: {n_clusters_silhouette} clusters (score={best_silhouette:.3f})")
    
    # 2.8 --- Return standardized output ---
    return {
        "data": None,
        "metadata": {
            "n_clusters_elbow": n_clusters_elbow,
            "n_clusters_silhouette": n_clusters_silhouette,
            "best_silhouette": best_silhouette,
            "n_variables": n_vars
        },
        "figures": {
            "dendrogram": str(dendro_path),
            "evaluation": str(eval_path)
        },
        "exports": None
    }


# ------------------------------------------------------------------------------------------------------------
# 3. Cattell Scree Plot with Kaiser Criterion
# ------------------------------------------------------------------------------------------------------------
def plot_cattell_criterion(eigenvalues: np.ndarray, explained_variance: np.ndarray) -> Tuple[plt.Figure, str]:
    """
    Generate Cattell scree plot with Kaiser criterion threshold (λ=1).

    Displays eigenvalues per component with horizontal line at λ=1 to identify
    components retaining more variance than a single standardized variable.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues from PCA (λ values).
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure
        - path (str): Saved file path
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 3.1 --- Plot eigenvalues ---
    ax.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        marker='o',
        linestyle='-',
        linewidth=1.8,
        markersize=6,
        label="Valeurs propres"
    )   # eigenvalue curve

    # 3.2 --- Kaiser criterion threshold ---
    ax.axhline(
        y=1,
        color='red',
        linestyle='--',
        linewidth=1.0,
        label='Seuil λ=1 (Cattell)'
    )   # threshold line

    # 3.3 --- Configure axes ---
    ax.set_xlabel("Composantes principales")   # x-axis label
    ax.set_ylabel("Valeurs propres")   # y-axis label
    ax.set_xticks(range(1, len(eigenvalues) + 1))   # x-axis ticks
    ax.legend(loc='upper right', frameon=False)   # legend without border

    # 3.4 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=False,   # legend already created
        main_title="Critère du coude de Cattell",
        main_title_size=16,
        tight_layout=True
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    plt.show()   # display

    # 3.5 --- Save figure ---
    output_path = OUTPUT_PATHS["figures"] / "msti_mfa_cattell.png"   # define path
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")   # DPI from rcParams
    print(f"Cattell plot saved to: {output_path}")   # confirmation

    return fig, str(output_path)


# ------------------------------------------------------------------------------------------------------------
# 4. MFA Initialization and Core Computation
# ------------------------------------------------------------------------------------------------------------
def _compute_variable_correlations(pca_model: PCA, data_std: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Compute proper correlations between original variables and principal components.
    
    For standardized data, the correlation between variable j and component k is:
        cor(X_j, F_k) = loading_jk * sqrt(λ_k)
    
    where:
        - loading_jk is the eigenvector coefficient
        - λ_k is the k-th eigenvalue
    
    Parameters
    ----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    data_std : pandas.DataFrame
        Standardized input data used for PCA.
    n_components : int, default=3
        Number of principal components to compute correlations for.
        
    Returns
    -------
    pandas.DataFrame
        Correlation matrix (variables × components) with proper formula.
    """
    # 4.1 --- Extract PCA components ---
    loadings = pca_model.components_[:n_components].T   # extract loadings (n_vars, n_components)
    eigenvalues = pca_model.explained_variance_[:n_components]   # extract eigenvalues
    
    # 4.2 --- Compute proper correlations ---
    correlations = loadings * np.sqrt(eigenvalues)   # proper correlation formula: loading × √λ
    
    # 4.3 --- Build correlation dataframe ---
    corr_df = pd.DataFrame(
        correlations,
        index=data_std.columns,
        columns=[f'F{i+1}' for i in range(n_components)]
    )   # create dataframe with variable names and component labels
    
    return corr_df


def run_mfa_projection(imputed: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Execute Multiple Factor Analysis (MFA) on MSTI indicators grouped by homogeneous blocks.

    MFA performs a two-stage PCA:
    1. Block-wise PCA with weighting (w_g = 1/√λ₁ per block)
    2. Global PCA on weighted block axes

    Outputs include:
    - Observation coordinates (F1–F3)
    - Variable–component correlations
    - Quality metrics (COS², contributions)
    - Inertia table with cumulative variance

    Global variables stored for downstream plotting:
    - obs_coords_df: Observation coordinates (F1–F3)
    - explained_variance: Variance explained per axis (%)
    - clean_data: MFA-structured data with MultiIndex columns
    - correlations_df: Variable-component correlations 

    Parameters
    ----------
    imputed : pandas.DataFrame
        Imputed and indexed dataset with MSTI MultiIndex:
        (zone, continent, decade, year) or equivalent compliant index.

    Returns
    -------
    tuple
        - exports_dict (dict): Paths to CSV exports and core arrays
        - figures_dict (dict): Empty placeholder for downstream visualizations
    """
    
    global obs_coords_df, explained_variance, clean_data, correlations_df   # three global variables for downstream visualization

    print("\n=== STEP: Multiple Factor Analysis ===")

    # 4.1 --- Extract numeric variables ---
    numeric = get_numeric_columns(imputed)   # select quantitative indicators

    # 4.2 --- Define variable blocks ---
    all_vars = sum(VARIABLE_BLOCKS.values(), [])   # flatten all block variables
    missing_vars = [v for v in all_vars if v not in numeric.columns]   # missing variables check
    if missing_vars:
        print(f"⚠ Missing variables: {missing_vars}")

    print(f"• MFA structure: {len(all_vars)} variables in {len(VARIABLE_BLOCKS)} blocks")

    # 4.3 --- Build MultiIndex (Block, Variable) ---
    multi_cols, selected_vars = [], []   # initialize structures
    for block_name, block_vars in VARIABLE_BLOCKS.items():
        for var in block_vars:
            if var in numeric.columns:
                multi_cols.append((block_name, var))
                selected_vars.append(var)

    clean_data = numeric[selected_vars].copy()   # retain valid variables
    clean_data.columns = pd.MultiIndex.from_tuples(multi_cols, names=["Block", "Variable"])   # set multiindex
    print(f"• Retained {clean_data.shape[1]} variables across {len(VARIABLE_BLOCKS)} thematic blocks")

    # 4.4 --- Block-wise PCA weighting ---
    print("\n=== Principal eigenvalues and relative weights per MFA block === ")
    block_weights, block_first_axes = {}, []
    for block_name in clean_data.columns.get_level_values("Block").unique():
        block_data = clean_data[block_name]
        pca_block = PCA(n_components=1)
        pca_block.fit(block_data)

        lambda1 = pca_block.explained_variance_[0]
        weight = 1.0 / np.sqrt(lambda1) if lambda1 > 0 else 1.0
        weight = min(weight, 3.0)   # cap extreme weights

        block_weights[block_name] = weight
        block_first_axes.append((block_name, pca_block.transform(block_data)[:, 0] * weight))
        print(f"  Block '{block_name}': λ1={lambda1:.3f}, weight={weight:.3f}")

    # 4.5 --- Global PCA on weighted block axes ---
    block_axes_df = pd.DataFrame({name: axis for name, axis in block_first_axes}, index=clean_data.index)
    pca_global = PCA(n_components=5)
    pca_global.fit(block_axes_df)

    # 4.6 --- Extract eigenvalues and variance ---
    eigenvalues = pca_global.explained_variance_
    explained_variance = pca_global.explained_variance_ratio_ * 100
    cumulative_inertia = np.cumsum(explained_variance)

    inertia_df = pd.DataFrame({
        "Component": [f"F{i+1}" for i in range(len(explained_variance))],
        "Inertia (%)": explained_variance,
        "Cumulative Inertia (%)": cumulative_inertia
    })

    print("\n=== Cumulative Inertia on Axes ===")
    print(inertia_df.round(2).to_string(index=False))
    print(f"\nCumulative inertia on 3 axes: {cumulative_inertia[min(2, len(cumulative_inertia) - 1)]:.1f}%")

    # 4.7 --- Export inertia table ---
    inertia_path = OUTPUT_PATHS["reports"] / "msti_mfa_inertia.csv"
    inertia_df.to_csv(inertia_path, index=False)
    print(f"\n✓ Inertia table exported → {inertia_path}")

    # 4.8 --- Observation coordinates (F1–F3) ---
    n_axes = min(3, pca_global.n_components_)
    obs_coords = pca_global.transform(block_axes_df)[:, :n_axes]
    axes_labels = [f"F{i+1}" for i in range(n_axes)]
    obs_coords_df = pd.DataFrame(obs_coords, columns=axes_labels, index=clean_data.index)

    # 4.9 --- Variable–component correlations ---
    correlations = _compute_variable_correlations(pca_global, block_axes_df, n_components=n_axes)

    # 4.10 --- Variable metrics: Contributions, COS² ---
    corr_values = correlations.values
    n_vars, n_axes = corr_values.shape

    # COS² (quality of representation) ---
    cos2 = corr_values ** 2
    cos2_df = pd.DataFrame(
        cos2,
        index=correlations.index,
        columns=[f"F{i+1}" for i in range(n_axes)]
    )
    cos2_df["Total_COS2"] = cos2_df.sum(axis=1)

    # Contributions (%) ---
    contrib = (cos2 / cos2.sum(axis=0)) * 100
    contrib_df = pd.DataFrame(
        contrib,
        index=correlations.index,
        columns=[f"F{i+1}_contrib(%)" for i in range(n_axes)]
    )
    contrib_df["Total_contrib(%)"] = contrib_df.sum(axis=1)

    # 4.11 --- Observation metrics (COS² & CTR) + Exports Excel ---
    obs_sq = obs_coords_df ** 2   # Square of coordinates per axis
    inertia_obs = obs_sq.sum(axis=1)   # Total inertia per observation
    obs_cos2 = obs_sq.div(inertia_obs, axis=0)   # COS² per axis
    obs_cos2["Total_COS2"] = obs_cos2.sum(axis=1)

    # 4.11.1 --- Combine and export ---
    metrics_df = pd.concat([cos2_df, contrib_df], axis=1)
    metrics_path = OUTPUT_PATHS["reports"] / "msti_mfa_variable_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"✓ Variable metrics exported → {metrics_path}")

    print("\n=== COS² (Quality of representation per variable) ===")
    print(cos2_df.round(4).to_string())
    print("\n=== Contributions (%) of variables to axes ===")
    print(contrib_df.round(2).to_string())

    print("\nComputing observation metrics by continent and country")
    
    # 4.11.2 --- Build obs_metrics ---
    obs_metrics = obs_coords_df.copy()
    obs_metrics['Continent'] = obs_coords_df.index.get_level_values('continent')
    obs_metrics['Country'] = obs_coords_df.index.get_level_values('zone')
    obs_metrics['Year'] = obs_coords_df.index.get_level_values('year')
    obs_metrics['Decade'] = obs_coords_df.index.get_level_values('decade')
    
    # 4.11.3 --- Add COS² per axis ---
    for i in range(n_axes):
        obs_metrics[f'F{i+1}_COS2'] = obs_cos2[f'F{i+1}']
    obs_metrics['Total_COS2'] = obs_cos2['Total_COS2']
    
    # 4.11.4 --- Add CTR (%) per axis ---
    obs_sq_sum = obs_sq.sum(axis=0)
    for i in range(n_axes):
        obs_metrics[f'F{i+1}_CTR(%)'] = (obs_sq[f'F{i+1}'] / obs_sq_sum[f'F{i+1}']) * 100
    
    # 4.11.5 --- Aggregate by country & continent ---
    agg_cols = []
    for i in range(n_axes):
        agg_cols.extend([f'F{i+1}', f'F{i+1}_COS2', f'F{i+1}_CTR(%)'])
    agg_cols.append('Total_COS2')
    
    country_metrics = obs_metrics.groupby('Country')[agg_cols].mean()
    country_metrics['Continent'] = obs_metrics.groupby('Country')['Continent'].first()
    
    ordered_cols = ['Continent']
    for i in range(n_axes):
        ordered_cols.extend([f'F{i+1}', f'F{i+1}_COS2', f'F{i+1}_CTR(%)'])
    ordered_cols.append('Total_COS2')
    country_metrics = country_metrics[ordered_cols]
    continent_metrics = obs_metrics.groupby('Continent')[agg_cols].mean()

    ordered_cols_continent = []
    for i in range(n_axes):
        ordered_cols_continent.extend([f'F{i+1}', f'F{i+1}_COS2', f'F{i+1}_CTR(%)'])
    ordered_cols_continent.append('Total_COS2')
    continent_metrics = continent_metrics[ordered_cols_continent]
    
    # 4.12 --- Export observation metrics to Excel ---
    excel_path = OUTPUT_PATHS["reports"] / "msti_mfa_obs_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        country_metrics.to_excel(writer, sheet_name='Countries', index=True)
        continent_metrics.to_excel(writer, sheet_name='Continents', index=True)
    print(f"✓ Observation metrics (Excel) exported → {excel_path}")
    print(f"  • Sheet 'Countries': {len(country_metrics)} countries")
    print(f"  • Sheet 'Continents': {len(continent_metrics)} continents")

    # 4.13 --- Display countries (sorted by Total COS²) ---
    print("\n=== Summary of MFA Observation Metrics ===")
    print("Countries — Quality of Representation (COS²) and Contributions (CTR)")
    
    top_countries = country_metrics.nlargest(10, 'Total_COS2')
    
    print("{:<4} {:<28} {:<11} | {:^30} | {:^30} | {:^30} | {:>10}".format(
        "Rank", "Country", "Continent",
        "F1 (Coord | COS² | CTR%)", "F2 (Coord | COS² | CTR%)", "F3 (Coord | COS² | CTR%)",
        "Total_COS²"
    ))
    print("-" * 160)
    
    for rank, (country, row) in enumerate(top_countries.iterrows(), 1):
        print(f"{rank:<4} {country:<28} {row['Continent']:<11} | "
              f"{row['F1']:7.4f} {row['F1_COS2']:7.4f} {row['F1_CTR(%)']:6.2f} | "
              f"{row['F2']:7.4f} {row['F2_COS2']:7.4f} {row['F2_CTR(%)']:6.2f} | "
              f"{row['F3']:7.4f} {row['F3_COS2']:7.4f} {row['F3_CTR(%)']:6.2f} | "
              f"{row['Total_COS2']:10.4f}")
    
    # 4.14 --- Display Continents (sorted by Total COS²) ---
    print("\nContinents — Quality of Representation (COS²) and Contributions (CTR)")

    top_continents = continent_metrics.nlargest(5, 'Total_COS2')
    
    print("{:<4} {:<16} | {:^30} | {:^30} | {:^30} | {:>10}".format(
        "Rank", "Continent",
        "F1 (Coord | COS² | CTR%)", "F2 (Coord | COS² | CTR%)", "F3 (Coord | COS² | CTR%)",
        "Total_COS²"
    ))
    print("-" * 140)
    
    for rank, (continent, row) in enumerate(top_continents.iterrows(), 1):
        print(f"{rank:<4} {continent:<16} | "
              f"{row['F1']:7.4f} {row['F1_COS2']:7.4f} {row['F1_CTR(%)']:6.2f} | "
              f"{row['F2']:7.4f} {row['F2_COS2']:7.4f} {row['F2_CTR(%)']:6.2f} | "
              f"{row['F3']:7.4f} {row['F3_COS2']:7.4f} {row['F3_CTR(%)']:6.2f} | "
              f"{row['Total_COS2']:10.4f}")
    
    # 4.15 --- Package exports for downstream visualization ---
    exports = {
        "inertia": str(inertia_path),
        "variable_metrics": str(metrics_path),
        "obs_metrics_excel": str(excel_path),
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance": explained_variance.tolist(),
        "axes_labels": axes_labels
    }

    # 4.16 --- Computing block-level indicators (Sparse PCA) ---
    print("\n=== Block-level Sparse PCA synthesis ===")
    blocks_df = pd.DataFrame(index=clean_data.index)   # initialize output dataframe
    spca_alpha = 0.1   # sparsity penalty (0.1 = moderate sparsity)
    spca_ridge = 0.01   # L2 regularization for numerical stability
    
    for block_name in clean_data.columns.get_level_values("Block").unique():
        block_data = clean_data[block_name]   # extract block variables
        n_vars = len(block_data.columns)   # count variables in block
        
        # 4.16.1 --- Fit Sparse PCA ---
        spca = SparsePCA(
            n_components=1,
            alpha=spca_alpha,
            ridge_alpha=spca_ridge,
            max_iter=100,
            random_state=DEFAULT_RANDOM_STATE
        )   # initialize sparse PCA model
        indicator = spca.fit_transform(block_data).ravel()   # compute sparse score
        blocks_df[block_name] = indicator   # store in output dataframe
        
        # 4.16.2 --- Compute diagnostics ---
        n_active = np.sum(np.abs(spca.components_[0]) > 1e-6)   # count non-zero loadings
        var_explained = np.var(indicator) / np.sum(np.var(block_data, axis=0))   # variance ratio
        
        print(f"  Block '{block_name}':")
        print(f"    • Active variables: {n_active}/{n_vars} ({n_active/n_vars*100:.1f}%)")
        print(f"    • Variance explained: {var_explained*100:.1f}%")

    # 4.17 --- Export Sparse PCA scores to CSV ---
    spca_scores = blocks_df.copy()   # copy sparse PCA scores dataframe
    output_path = DATA_PATHS["processed"] / "msti_spca_final.csv"   # define output path
    spca_scores.to_csv(output_path, sep=";")   # export with semicolon separator
    exports["mfa_final_csv"] = str(output_path)   # store path in exports dict
    
    print(f"\n✓ Sparse PCA block-level dataset exported → {output_path}")
    print(f"  • Shape: {spca_scores.shape[0]} observations × {spca_scores.shape[1]} blocks")
    print(f"  • Index: {MSTI_INDEX_LABELS}")
    print(f"  • Blocks: {list(spca_scores.columns)}")

    # Store key artifacts in global scope (for interactive plotting)
    globals().update({
        "obs_coords_df": obs_coords_df,
        "explained_variance": explained_variance,
        "clean_data": clean_data,
        "correlations_df": correlations
    })

    return exports, {}


# ------------------------------------------------------------------------------------------------------------
# 5. Correlation Circles
# ------------------------------------------------------------------------------------------------------------
def plot_mfa_correlation_circle_f12(
    correlations_df: pd.DataFrame,
    explained_variance: np.ndarray,
    title: str = "MFA Correlation Circle (F1–F2)",
    filename: str = "msti_mfa_correlation_circle_f12.png",
) -> Tuple[plt.Figure, str]:
    """
    Plot the correlation circle for MFA components F1–F2.

    Displays each variable as an arrow from the origin according to its
    correlation coefficients with F1 and F2, including a unit circle.

    Parameters
    ----------
    correlations_df : pandas.DataFrame
        Variable-component correlations with F1 and F2 columns.
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).
    title : str, default="MFA Correlation Circle (F1–F2)"
        Plot title.
    filename : str, default="msti_mfa_correlation_circle_f12.png"
        Output filename.

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure
        - path (str): Saved file path
    """
    # 5.1 --- Check required columns ---
    if not {"F1", "F2"}.issubset(correlations_df.columns):
        raise ValueError("DataFrame must contain columns 'F1' and 'F2'.")

    # 5.2 --- Create figure ---
    fig, ax = plt.subplots(figsize=(10, 10))   # square for circle

    # 5.3 --- Draw unit circle ---
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--", linewidth=1)
    ax.add_artist(circle)

    # 5.4 --- Draw variable vectors ---
    for var, (x, y) in correlations_df[["F1", "F2"]].iterrows():
        ax.arrow(0, 0, x, y,
                 alpha=0.75, head_width=0.02, length_includes_head=True, color="#333333")
        txt = ax.text(x * 1.07, y * 1.07, var, fontsize=7, ha="center", va="center", color="black")
        txt.set_path_effects([withStroke(linewidth=2, foreground="white")])   # halo

    # 5.5 --- Configure axes ---
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")   # horizontal axis
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")   # vertical axis
    ax.set_xlabel(f"F1 ({explained_variance[0]:.1f}%)")   # x-axis
    ax.set_ylabel(f"F2 ({explained_variance[1]:.1f}%)")   # y-axis
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="datalim")

    # 5.6 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=False,
        main_title=title,
        main_title_size=16,
        tight_layout=True
    )

    plt.show()

    # 5.7 --- Save output ---
    out_path = OUTPUT_PATHS["figures"] / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")   # DPI from rcParams
    print(f"Correlation circle (F1–F2) saved to: {out_path}")

    return fig, str(out_path)


def plot_mfa_correlation_circle_f13(
    correlations_df: pd.DataFrame,
    explained_variance: np.ndarray,
    title: str = "MFA Correlation Circle (F1–F3)",
    filename: str = "msti_mfa_correlation_circle_f13.png",
) -> Tuple[plt.Figure, str]:
    """
    Plot the correlation circle for MFA components F1–F3.

    Same logic as F1–F2 correlation circle, but using the third axis.

    Parameters
    ----------
    correlations_df : pandas.DataFrame
        Variable-component correlations with F1 and F3 columns.
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).
    title : str, default="MFA Correlation Circle (F1–F3)"
        Plot title.
    filename : str, default="msti_mfa_correlation_circle_f13.png"
        Output filename.

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure
        - path (str): Saved file path
    """
    # 5.1 --- Check required columns ---
    if not {"F1", "F3"}.issubset(correlations_df.columns):
        raise ValueError("DataFrame must contain columns 'F1' and 'F3'.")

    # 5.2 --- Create figure ---
    fig, ax = plt.subplots(figsize=(10, 10))   # square for circle

    # 5.3 --- Draw unit circle ---
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--", linewidth=1)
    ax.add_artist(circle)

    # 5.4 --- Draw variable vectors ---
    for var, (x, y) in correlations_df[["F1", "F3"]].iterrows():
        ax.arrow(0, 0, x, y,
                 alpha=0.75, head_width=0.02, length_includes_head=True, color="#333333")
        txt = ax.text(x * 1.07, y * 1.07, var, fontsize=7, ha="center", va="center", color="black")
        txt.set_path_effects([withStroke(linewidth=2, foreground="white")])   # halo
    
    # 5.5 --- Configure axes ---
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel(f"F1 ({explained_variance[0]:.1f}%)")
    ax.set_ylabel(f"F3 ({explained_variance[min(2, len(explained_variance)-1)]:.1f}%)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="datalim")

    # 5.6 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=False,
        main_title=title,
        main_title_size=16,
        tight_layout=True
    )

    plt.show()

    # 5.7 --- Save output ---
    out_path = OUTPUT_PATHS["figures"] / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")   # DPI from rcParams
    print(f"Correlation circle (F1–F3) saved to: {out_path}")

    return fig, str(out_path)


# ------------------------------------------------------------------------------------------------------------
# 6. Observation Projections
# ------------------------------------------------------------------------------------------------------------
def plot_mfa_projection_continents_f12(
    obs_coords_df: pd.DataFrame,
    explained_variance: np.ndarray,
    clean_data: pd.DataFrame,
    spline_s: float = 2.0,
    title: str = "MFA 2D Projection — Spline barycenters by continent (F1–F2)",
    filename: str = "msti_mfa_projection_continents_f12.png",
) -> Tuple[plt.Figure, str]:
    """
    Plot MFA observation projection on F1–F2 using spline-based temporal barycenters.

    Each continent is represented by the smoothed trajectory of its barycenters
    across decades, computed using spline interpolation rather than simple means.
    This highlights long-term structural shifts while preserving the MFA geometry.

    Parameters
    ----------
    obs_coords_df : pandas.DataFrame
        MFA coordinates with F1 and F2 columns.
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).
    clean_data : pandas.DataFrame
        Original dataset with MultiIndex (Zone, Continent, Year, Decade).
    spline_s : float, default=2.0
        Smoothing parameter for spline interpolation (higher = smoother).
    title : str
        Plot title.
    filename : str
        Output filename.

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure.
        - path (str): Saved file path.

    Notes
    -----
    Countries with fewer than 3 temporal observations are excluded to prevent
    spline instability. Ellipses represent 95% confidence regions per continent.
    """
    # 6.1 --- Prepare data ---
    continents, decades, alpha_dict = extract_graphics_params(clean_data)
    color_map = PALETTE_CONTINENT  # load color palette

    proj_df = pd.DataFrame({
        "Continent": continents,
        "Decade": decades,
        "F1": obs_coords_df["F1"],
        "F2": obs_coords_df["F2"]
    }, index=clean_data.index).reset_index(drop=True)

    # 6.2 --- Compute spline-based barycenters ---
    barycenters = compute_spline_barycenters(
        proj_df,
        axes=["F1", "F2"],
        time_col="Decade",
        group_col="Continent",
        spline_s=spline_s,
    )

    # 6.3 --- Static visualization ---
    fig, ax = plt.subplots(figsize=(16, 8))
    text_labels = []

    for continent, group in barycenters.groupby("Continent"):
        # 6.3.1 --- Draw 95% confidence ellipse ---
        draw_confidence_ellipse(
            group["F1"].values,
            group["F2"].values,
            ax=ax,
            edgecolor=color_map.get(continent, "gray"),
            alpha=0.8,
            linewidth=1.2
        )

        # 6.3.2 --- Scatter spline barycenters ---
        ax.scatter(
            group["F1"],
            group["F2"],
            s=180,
            color=color_map.get(continent, "gray"),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=continent
        )

        # 6.3.3 --- Add colored labels per continent ---
        txt = ax.text(
            group["F1"].mean(),
            group["F2"].mean(),
            continent,
            fontsize=10,
            ha="center",
            va="center",
            color=color_map.get(continent, "gray"),
            alpha=0.95,
            fontweight="bold"
        )
        txt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])
        text_labels.append(txt)

    # 6.3.4 --- Adjust text placement ---
    adjust_text(text_labels, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5), ax=ax)

    # 6.4 --- Axes configuration ---
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel(f"F1 ({explained_variance[0]:.1f}%)")
    ax.set_ylabel(f"F2 ({explained_variance[1]:.1f}%)")

    # 6.5 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=True,
        color_map=color_map,
        legend_title=f"Continents (spline barycenters, smoothing={spline_s:.1f})",
        main_title=title,
        main_title_size=16,
        tight_layout=True
    )

    # 6.6 --- Save figure ---
    out_path = OUTPUT_PATHS["figures"] / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")

    return fig, str(out_path)


def plot_mfa_projection_continents_f13(
    obs_coords_df: pd.DataFrame,
    explained_variance: np.ndarray,
    clean_data: pd.DataFrame,        # 🔹 à ajouter ici
    spline_s: float = 2.0,
    title: str = "MFA 2D Projection – Spline by continent and decade (F1–F3)",
    filename: str = "msti_mfa_projection_continents_f13.png",
) -> Tuple[plt.Figure, str]:
    """
    Plot MFA observation projection on F1–F3 aggregated by continent and decade.

    Same logic as F1–F3 projection but using the third principal component.

    Parameters
    ----------
    obs_coords_df : pandas.DataFrame
        MFA coordinates with F1 and F3 columns.
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).
    clean_data : pandas.DataFrame
        Original indexed data with zone, continent, decade information.
    title : str, default="MFA 2D Projection – Spline by continent and decade (F1–F3)"
        Plot title.
    filename : str, default="msti_mfa_projection_continents_f13.png"
        Output filename.

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure
        - path (str): Saved file path
    """
    # 7.1 --- Prepare data ---
    continents, decades, alpha_dict = extract_graphics_params(clean_data)
    color_map = PALETTE_CONTINENT  # load color palette

    proj_df = pd.DataFrame({
        "Continent": continents,
        "Decade": decades,
        "F1": obs_coords_df["F1"],
        "F3": obs_coords_df["F3"]
    }, index=clean_data.index).reset_index(drop=True)

    # 7.2 --- Compute spline-based barycenters ---
    barycenters = compute_spline_barycenters(
        proj_df,
        axes=["F1", "F3"],
        time_col="Decade",
        group_col="Continent",
        spline_s=spline_s,
    )

    # 7.3 --- Static visualization ---
    fig, ax = plt.subplots(figsize=(16, 8))
    text_labels = []

    for continent, group in barycenters.groupby("Continent"):
        # 7.3.1 --- Draw 95% confidence ellipse ---
        draw_confidence_ellipse(
            group["F1"].values,
            group["F3"].values,
            ax=ax,
            edgecolor=color_map.get(continent, "gray"),
            alpha=0.8,
            linewidth=1.2
        )

        # 7.3.2 --- Scatter spline barycenters ---
        ax.scatter(
            group["F1"],
            group["F3"],
            s=180,
            color=color_map.get(continent, "gray"),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=continent
        )

        # 7.3.3 --- Add colored labels per continent ---
        txt = ax.text(
            group["F1"].mean(),
            group["F3"].mean(),
            continent,
            fontsize=10,
            ha="center",
            va="center",
            color=color_map.get(continent, "gray"),
            alpha=0.95,
            fontweight="bold"
        )
        txt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])
        text_labels.append(txt)

    # 7.3.4 --- Adjust text placement ---
    adjust_text(text_labels, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5), ax=ax)

    # 7.4 --- Axes configuration ---
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel(f"F1 ({explained_variance[0]:.1f}%)")
    ax.set_ylabel(f"F3 ({explained_variance[1]:.1f}%)")

    # 7.5 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=True,
        color_map=color_map,
        legend_title=f"Continents (spline barycenters, smoothing={spline_s:.1f})",
        main_title=title,
        main_title_size=16,
        tight_layout=True
    )

    # 7.6 --- Save figure ---
    out_path = OUTPUT_PATHS["figures"] / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")

    return fig, str(out_path)


def plot_mfa_projection_countries_f12(
    obs_coords_df: pd.DataFrame,
    explained_variance: np.ndarray,
    clean_data: pd.DataFrame,
    spline_s: float = 2.0,
    title: str = "MFA 2D Projection — Spline barycenters by country (F1–F2)",
    filename: str = "msti_mfa_projection_countries_f12.png",
) -> Tuple[plt.Figure, str]:
    """
    Plot MFA observation projection on F1–F2 using spline-based temporal barycenters.

    Each country is represented by a single point computed as the center of gravity
    of its smoothed temporal trajectory. This reduces label clutter while preserving
    the overall geometric structure of the MFA space.

    Parameters
    ----------
    obs_coords_df : pandas.DataFrame
        MFA coordinates with F1 and F2 columns.
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).
    clean_data : pandas.DataFrame
        Original indexed data with zone, continent, decade information.
    spline_s : float, default=2.0
        Smoothing parameter for temporal splines (higher = smoother, more stable).
    title : str
        Plot title.
    filename : str
        Output filename.

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure
        - path (str): Saved file path
    
    Notes
    -----
    Countries with fewer than 3 temporal observations are excluded to prevent
    spline instability. Extreme outliers (>10x IQR) are clipped to avoid scale issues.
    """
    # 8.1 --- Prepare projection dataframe ---
    continents, decades, _ = extract_graphics_params(clean_data)
    zones = clean_data.index.get_level_values("zone")

    proj_df = pd.DataFrame({
        "Zone": zones,
        "Continent": continents,
        "Decade": decades,
        "F1": obs_coords_df["F1"],
        "F2": obs_coords_df["F2"],
    })

    # 8.2 --- Compute spline barycenters ---
    barycenters = compute_spline_barycenters(
        proj_df,
        axes=["F1", "F2"],
        time_col="Decade",
        group_col="Zone",
        spline_s=spline_s,
        n_points=200
    )   # returns DataFrame with Zone as index

    # 8.2.1 --- Clip extreme outliers (protection against scale aberrations) ---
    for ax in ["F1", "F2"]:
        q75 = barycenters[ax].abs().quantile(0.75)
        threshold = 10 * q75
        n_before = len(barycenters)
        barycenters = barycenters[barycenters[ax].abs() <= threshold]
        n_clipped = n_before - len(barycenters)

    # 8.2.2 --- Add continent metadata (CORRECTION: Zone est maintenant l'index) ---
    zone_to_continent = dict(zip(zones, continents))   # create mapping
    barycenters["Continent"] = barycenters.index.map(zone_to_continent)   # map using index

    # 8.3 --- Static visualization: F1–F2 projection with continent-colored labels ---
    fig, ax = plt.subplots(figsize=(16, 8))
    text_labels = []  # initialize text label list

    # 8.3.1 --- Map countries to continents ---
    barycenters["Continent"] = barycenters.index.map(COUNTRY_TO_CONTINENT)  # map index (country) to continent
    color_map = PALETTE_CONTINENT  # standardized continent color palette

    # 8.3.2 --- Scatter points and text labels by continent ---
    for continent, group in barycenters.groupby("Continent"):  # iterate over continents
        draw_confidence_ellipse(
            group["F1"].values,
            group["F2"].values,
            ax=ax,
            edgecolor=color_map.get(continent, "gray"),
            alpha=0.7,
            linewidth=1.0
        )  # draw 95% confidence ellipse around country barycenters
        ax.scatter(
            group["F1"],
            group["F2"],
            s=120,
            color=color_map.get(continent, "gray"),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8
        )  # plot barycenters for each continent

        for _, row in group.iterrows():  # iterate over countries within the continent
            txt = ax.text(
                row["F1"],
                row["F2"],
                row.name,  # country name from index
                fontsize=7,
                ha="center",
                va="center",
                color=color_map.get(continent, "gray"),
                alpha=0.95
            )  # add label
            txt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])  # halo for readability
            text_labels.append(txt)  # store for later adjustment

    # 8.3.3 --- Automatic label adjustment ---
    adjust_text(
        text_labels,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        ax=ax
    )  # prevent overlapping labels

    # 8.4 --- Configure axes ---
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")  # horizontal reference line
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")  # vertical reference line
    ax.set_xlabel(f"F1 ({explained_variance[0]:.1f}%)")  # x-axis label
    ax.set_ylabel(f"F2 ({explained_variance[1]:.1f}%)")  # y-axis label

    # 8.5 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=True,
        color_map=PALETTE_COUNTRY, 
        legend_title=f"Countries (spline barycenters, smoothing={spline_s:.1f})",
        main_title=title,
        main_title_size=16,
        tight_layout=True
    )

    plt.show()

    # 8.7 --- Save figure ---
    out_path = OUTPUT_PATHS["figures"] / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    return fig, str(out_path)


def plot_mfa_projection_countries_f13(
    obs_coords_df: pd.DataFrame,
    explained_variance: np.ndarray,
    clean_data: pd.DataFrame,
    spline_s: float = 2.0,
    title: str = "MFA 2D Projection — Spline barycenters by country (F1–F3)",
    filename: str = "msti_mfa_projection_countries_f13.png",
) -> Tuple[plt.Figure, str]:
    """
    Plot MFA observation projection on F1–F3 using spline-based temporal barycenters.

    Same logic as F1–F2 but using the third principal component. Each country is
    represented by a single smoothed barycenter to improve readability.

    Parameters
    ----------
    obs_coords_df : pandas.DataFrame
        MFA coordinates with F1 and F3 columns.
    explained_variance : numpy.ndarray
        Explained variance ratio per component (%).
    clean_data : pandas.DataFrame
        Original indexed data with zone, continent, decade information.
    spline_s : float, default=2.0
        Smoothing parameter for temporal splines.
    title : str
        Plot title.
    filename : str
        Output filename.

    Returns
    -------
    tuple
        - fig (matplotlib.figure.Figure): Generated figure
        - path (str): Saved file path
    """
    # 9.1 --- Prepare projection dataframe ---
    continents, decades, _ = extract_graphics_params(clean_data)
    zones = clean_data.index.get_level_values("zone")

    proj_df = pd.DataFrame({
        "Zone": zones,
        "Continent": continents,
        "Decade": decades,
        "F1": obs_coords_df["F1"],
        "F3": obs_coords_df["F3"],
    })

    # 9.2 --- Compute spline barycenters ---
    barycenters = compute_spline_barycenters(
        proj_df,
        axes=["F1", "F3"],
        time_col="Decade",
        group_col="Zone",
        spline_s=spline_s,
        n_points=200
    )

    # 9.2.1 --- Clip extreme outliers ---
    for ax in ["F1", "F3"]:
        q75 = barycenters[ax].abs().quantile(0.75)
        threshold = 10 * q75
        n_before = len(barycenters)
        barycenters = barycenters[barycenters[ax].abs() <= threshold]
        n_clipped = n_before - len(barycenters)
 
    # 9.2.2 --- Add continent metadata ---
    zone_to_continent = dict(zip(zones, continents))
    barycenters["Continent"] = barycenters.index.map(zone_to_continent)

    # 9.3 --- Static visualization: F1–F3 projection with continent-colored labels ---
    fig, ax = plt.subplots(figsize=(16, 8))
    text_labels = []  # initialize text label list

    # 9.3.1 --- Map countries to continents ---
    barycenters["Continent"] = barycenters.index.map(COUNTRY_TO_CONTINENT)  # map index (country) to continent
    color_map = PALETTE_CONTINENT  # standardized continent color palette

    # 9.3.2 --- Scatter points and text labels by continent ---
    for continent, group in barycenters.groupby("Continent"):  # iterate over continents
        draw_confidence_ellipse(
            group["F1"].values,
            group["F3"].values,
            ax=ax,
            edgecolor=color_map.get(continent, "gray"),
            alpha=0.7,
            linewidth=1.0
        )  # draw 95% confidence ellipse around country barycenters
        ax.scatter(
            group["F1"],
            group["F3"],
            s=120,
            color=color_map.get(continent, "gray"),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8
        )  # plot barycenters for each continent

        for _, row in group.iterrows():  # iterate over countries within the continent
            txt = ax.text(
                row["F1"],
                row["F3"],
                row.name,  # country name from index
                fontsize=7,
                ha="center",
                va="center",
                color=color_map.get(continent, "gray"),
                alpha=0.95
            )  # add label
            txt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])  # halo for readability
            text_labels.append(txt)  # store for later adjustment

    # 9.3.3 --- Automatic label adjustment (same as continent-level plots) ---
    adjust_text(
        text_labels,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        ax=ax
    )  # prevent overlapping labels

    # 9.4 --- Configure axes ---
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")  # horizontal reference line
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")  # vertical reference line
    ax.set_xlabel(f"F1 ({explained_variance[0]:.1f}%)")  # x-axis label
    ax.set_ylabel(f"F3 ({explained_variance[1]:.1f}%)")  # y-axis label

    # 9.5 --- Apply unified style ---
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=True,
        color_map=PALETTE_COUNTRY,
        legend_title=f"Countries (spline barycenters, smoothing={spline_s:.1f})",
        main_title=title,
        main_title_size=16,
        tight_layout=True
    )

    plt.show()

    # 9.7 --- Save figure ---
    out_path = OUTPUT_PATHS["figures"] / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    return fig, str(out_path)


# ============================================================================
# 10. Standalone Module Test
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MSTI CAH + MFA ANALYSIS - STANDALONE TEST")
    print("=" * 80)

    # 10.1 --- Load imputed dataset ---
    imputed_path = DATA_PATHS["processed"] / "msti_imputed.csv"
    print("\n[1/5] Loading imputed dataset...")
    imputed = pd.read_csv(imputed_path, sep=";", index_col=[0, 1, 2, 3])
    print(f"    ✓ Loaded: {imputed.shape}")

    # 10.2 --- Run hierarchical clustering analysis ---
    print("\n[2/5] Running hierarchical clustering (CAH)...")
    print()
    cah_result = cluster_variables_hierarchical(imputed)
    print(f"    ✓ Elbow clusters: {cah_result['metadata']['n_clusters_elbow']}")
    print(f"    ✓ Silhouette clusters: {cah_result['metadata']['n_clusters_silhouette']}")
    print(f"    ✓ Best silhouette: {cah_result['metadata']['best_silhouette']:.3f}")
    print(f"    ✓ Dendrogram saved: {cah_result['figures']['dendrogram']}")
    print(f"    ✓ Evaluation saved: {cah_result['figures']['evaluation']}")

    # 10.3 --- Run multiple factor analysis with integrated outputs ---
    print("\n[3/5] Running multiple factor analysis (MFA)...")
    print()
    mfa_exports, _ = run_mfa_projection(imputed)
    
    fig_cattell, cattell_path = plot_cattell_criterion(
        np.array(mfa_exports["eigenvalues"]),
        np.array(mfa_exports["explained_variance"])
    )
    print(f"    ✓ Cattell plot saved: {cattell_path}")
    print(f"    ✓ Variable metrics exported: {mfa_exports['variable_metrics']}")
    print(f"    ✓ Sparse PCA block-level dataset exported: {mfa_exports['mfa_final_csv']}")

    # 10.3.1 --- Display Sparse PCA diagnostics ---
    print("\n[3.1] Sparse PCA diagnostics per block:")
    spca_final = pd.read_csv(mfa_exports['mfa_final_csv'], sep=";", index_col=[0, 1, 2, 3])   # load final dataset
    for block_name in spca_final.columns:
        score_min = spca_final[block_name].min()   # minimum score
        score_max = spca_final[block_name].max()   # maximum score
        score_mean = spca_final[block_name].mean()   # mean score
        score_std = spca_final[block_name].std()   # standard deviation
        print(f"    • {block_name}:")
        print(f"        Range: [{score_min:.2f}, {score_max:.2f}]")
        print(f"        Mean: {score_mean:.2f}, Std: {score_std:.2f}")
    
    # 10.3.2 --- Retrieve global variables ---
    obs_coords_df = globals()["obs_coords_df"]   # MFA observation coordinates
    explained_variance = np.array(globals()["explained_variance"])   # variance per component
    clean_data = globals()["clean_data"]   # original multiindex data
    correlations_df = globals()["correlations_df"]   # variable-component correlations

    # 10.3.2 --- Generate correlation circles ---
    print("\n[3.1] Generating correlation circles...")
    fig_corr_f12, corr_f12_path = plot_mfa_correlation_circle_f12(
        correlations_df,
        explained_variance
    )
    fig_corr_f13, corr_f13_path = plot_mfa_correlation_circle_f13(
        correlations_df,
        explained_variance
    )
    print(f"    ✓ Correlation circle (F1×F2): {corr_f12_path}")
    print(f"    ✓ Correlation circle (F1×F3): {corr_f13_path}")

    # 10.4 --- Generate continent projections on factorial planes ---
    print("\n[4/5] Generating continent projections...")

    fig_cont_f12, cont_f12_path = plot_mfa_projection_continents_f12(
        obs_coords_df, 
        explained_variance, 
        clean_data
    )
    fig_cont_f13, cont_f13_path = plot_mfa_projection_continents_f13(
        obs_coords_df, 
        explained_variance, 
        clean_data
    )
    print(f"    ✓ Continent projection (F1×F2): {cont_f12_path}")
    print(f"    ✓ Continent projection (F1×F3): {cont_f13_path}")

    # 10.5 --- Generate country projections with spline barycenters ---
    print("\n[5/5] Generating country projections with spline barycenters...")
    spline_smoothing = 2.0
    fig_country_f12, country_f12_path = plot_mfa_projection_countries_f12(
        obs_coords_df, 
        explained_variance, 
        clean_data,
        spline_s=spline_smoothing
    )
    fig_country_f13, country_f13_path = plot_mfa_projection_countries_f13(
        obs_coords_df, 
        explained_variance, 
        clean_data,
        spline_s=spline_smoothing
    )
    print(f"    ✓ Country projection (F1×F2, spline): {country_f12_path}")
    print(f"    ✓ Country projection (F1×F3, spline): {country_f13_path}")

    # 10.6 --- Display export summary ---
    print("\n✓ EXPORT SUMMARY:")
    print(f"    • Dendrogram → {cah_result['figures']['dendrogram']}")
    print(f"    • CAH evaluation → {cah_result['figures']['evaluation']}")
    print(f"    • Cattell scree plot → {cattell_path}")
    print(f"    • Correlation circle (F1×F2) → {corr_f12_path}")
    print(f"    • Correlation circle (F1×F3) → {corr_f13_path}")  
    print(f"    • Inertia table → {mfa_exports['inertia']}")
    print(f"    • Variable metrics → {mfa_exports['variable_metrics']}")
    print(f"    • Observation metrics (Excel) → {mfa_exports['obs_metrics_excel']}")
    print(f"    • Sparse PCA dataset → {mfa_exports['mfa_final_csv']}")
    print(f"    • Continent projection (F1×F2) → {cont_f12_path}")
    print(f"    • Continent projection (F1×F3) → {cont_f13_path}")
    print(f"    • Country projection (F1×F2, spline) → {country_f12_path}")
    print(f"    • Country projection (F1×F3, spline) → {country_f13_path}")

    print("\n" + "=" * 80)
    print("✓ STANDALONE TEST COMPLETE")
    print("=" * 80)