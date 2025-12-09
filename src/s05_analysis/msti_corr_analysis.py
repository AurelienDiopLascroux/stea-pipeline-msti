# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/analysis/msti_corr_analysis.py
# PURPOSE      : Visualize correlation structure of quantitative MSTI variables
# DESCRIPTION  :
#   - Compute Pearson correlation coefficients between numerical variables.
#   - Display annotated heatmap using Matplotlib/Seaborn.
#   - Export correlation matrix to CSV for further analysis.
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
from typing import Dict   # type hints for return dictionaries

# 1.2 --- Third-party libraries ---
import matplotlib.pyplot as plt   # figure creation
import numpy as np   # numerical operations and masking
import pandas as pd   # dataframe operations
import seaborn as sns   # heatmap visualization

# 1.3 --- Internal modules ---
from src.config.msti_constants import get_numeric_columns
from src.config.msti_graphics_utils import standard_plot_style
from src.config.msti_paths_config import DATA_PATHS, OUTPUT_PATHS


# ------------------------------------------------------------------------------------------------------------
# 2. Main Functions
# ------------------------------------------------------------------------------------------------------------
def plot_correlation_matrix(
    imputed: pd.DataFrame,
    title: str = "MSTI Correlation Matrix",
    mask_upper: bool = True
) -> Dict:
    """
    Compute and visualize the Pearson correlation matrix of quantitative 
    MSTI variables.

    This function computes pairwise Pearson correlations between all 
    quantitative indicators, displays an annotated heatmap, and exports
    both the correlation matrix and the corresponding figure.

    Parameters
    ----------
    imputed : pandas.DataFrame
        Imputed dataset containing numerical MSTI indicators.
    title : str, default="MSTI Correlation Matrix"
        Title displayed above the heatmap.
    mask_upper : bool, default=True
        If True, hides the upper triangle of the correlation matrix.

    Returns
    -------
    dict
        Dictionary containing:
        - "data" : pandas.DataFrame
            The computed correlation matrix.
        - "metadata" : dict
            Shape, number of variables, and mean absolute correlation.
        - "figures" : dict
            Path of the saved PNG figure.
        - "exports" : str
            Path of the exported CSV file.
    """
    # 2. --- Display analysis header ---
    print("\n=== STEP: Correlation Matrix Analysis ===")

    # 3. --- Extract quantitative variables ---
    numeric = get_numeric_columns(imputed)
    n_vars = len(numeric.columns)
    print(f"Computing correlations for {n_vars} variables")

    # 4. --- Compute Pearson correlation matrix ---
    corr_matrix = numeric.corr(method="pearson")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if mask_upper else None

    # 5. --- Configure visualization parameters ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 160,
        "figure.dpi": 130,
        "axes.grid": False,
    })

    MAX_VARS_DISPLAY = 90   # maximum variables displayed in heatmap to avoid oversized figures
    
    fig_size = 0.4 * min(n_vars, MAX_VARS_DISPLAY)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # 6. --- Generate correlation heatmap ---
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7.5, "weight": "normal"},
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.25,
        linecolor="lightgray",
        square=True,
        cbar_kws={"shrink": 0.55, "label": "Pearson correlation"},
        ax=ax
    )

    # 7. --- Configure labels and title ---
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)

    # 8. --- Apply unified visual style ---
    standard_plot_style(
        fig=fig,
        grid=False,
        legend=False,
        main_title=None,
        tight_layout=False
    )

    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.92)

    # 9. --- Export figure and correlation matrix ---
    fig_path = OUTPUT_PATHS["figures"] / "msti_correlation_matrix.png"
    csv_path = OUTPUT_PATHS["reports"] / "msti_correlation_matrix.csv"

    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    corr_matrix.to_csv(csv_path, index=True)

    mean_abs_corr = corr_matrix.abs().mean().mean()
    print(f"Mean absolute correlation: {mean_abs_corr:.3f}")
    print(f"Correlation heatmap saved to: {fig_path}")
    print(f"Correlation matrix exported to: {csv_path}")

    plt.show()

    # 10. --- Return structured results dictionary ---
    return {
        "data": corr_matrix,
        "metadata": {
            "shape": corr_matrix.shape,
            "n_variables": n_vars,
            "mean_abs_correlation": mean_abs_corr
        },
        "figures": {"heatmap": str(fig_path)},
        "exports": str(csv_path)
    }


# ------------------------------------------------------------------------------------------------------------
# 3. Main Guard / Standalone Test
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MSTI CORRELATION ANALYSIS - STANDALONE TEST")
    print("=" * 60)

    # 3.1 --- Load imputed dataset ---
    imputed_path = DATA_PATHS["processed"] / "msti_imputed.csv"
    print("\n[1/2] Loading imputed dataset...")
    imputed = pd.read_csv(imputed_path, sep=";", index_col=[0, 1, 2, 3])
    print(f"    ✓ Loaded: {imputed.shape}")

    # 3.2 --- Run correlation analysis ---
    print("\n[2/2] Running correlation analysis...")
    result = plot_correlation_matrix(imputed)
    print(f"    ✓ Matrix shape: {result['metadata']['shape']}")
    print(f"    ✓ Mean |correlation|: {result['metadata']['mean_abs_correlation']:.3f}")
    print(f"    ✓ Figure saved: {result['figures']['heatmap']}")
    print(f"    ✓ CSV exported: {result['exports']}")

    print("\n" + "=" * 60)
    print("✓ STANDALONE TEST COMPLETE")
    print("=" * 60)
