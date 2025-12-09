# ============================================================================================================
# PROJECT      : STEA — Science, Technology & Energy Analysis
# PIPELINE     : MSTI — Main Science and Technology Indicators
# MODULE       : pipelines/analysis/msti_analysis_univariate.py
# PURPOSE      : Descriptive univariate analysis on MSTI dataset
# DESCRIPTION  :
#   - Generate descriptive statistics for quantitative variables
#   - Visual exploration with boxplots grouped by thematic blocks
#   - Continent-based outlier coloring for geographical pattern identification
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
from typing import Dict   # type hints

# 1.2 --- Third-party libraries ---
import matplotlib.pyplot as plt   # static visualization
import pandas as pd   # dataframe manipulation

# 1.3 --- Internal modules ---
from src.config.msti_constants import get_numeric_columns
from src.config.msti_graphics_utils import PALETTE_CONTINENT, standard_plot_style
from src.config.msti_paths_config import DATA_PATHS, OUTPUT_PATHS
from src.config.msti_variables_mapping import (
    COUNTRY_TO_CONTINENT,
    VARIABLE_BLOCKS
) # plotting style


# ------------------------------------------------------------------------------------------------------------
# 2. Univariate Descriptive Analysis
# ------------------------------------------------------------------------------------------------------------

def describe_univariate(imputed: pd.DataFrame) -> Dict:
    """
    Generate summary table of univariate descriptive statistics on 
    z-scored variables.

    Computes measures of central tendency (mean, median) and dispersion
    (std, min, max, quartiles) for all numeric MSTI variables.

    Parameters
    ----------
    imputed : pandas.DataFrame
        Imputed MSTI dataset with MultiIndex.
        
    Returns
    -------
    dict
        - data (pd.DataFrame): Descriptive statistics table
        - metadata (dict): Number of variables analyzed
        - exports (str): Path to saved CSV

    Notes
    -----
    All numeric columns are previously z-scored (mean=0, std=1).
    Interpretation focuses on relative dispersion rather than raw magnitudes.
    """
    print("\n=== STEP : Univariate Statistical Description ===")

    # 2.1 --- Extract numeric columns ---
    numeric = get_numeric_columns(imputed)
    
    # 2.2 --- Compute descriptive statistics ---
    stats = numeric.describe().round(2)
    n_vars = len(numeric.columns)
    
    # 2.3 --- Export to CSV ---
    output_path = OUTPUT_PATHS["reports"] / "msti_univariate_statistics.csv"
    stats.to_csv(output_path, index=True)
    
    print(f"Exported {n_vars} variable statistics → {output_path}")

    # 2.4 --- Return structured output ---
    return {
        "data": stats,
        "metadata": {"n_variables": n_vars},
        "exports": str(output_path)
    }


# ------------------------------------------------------------------------------------------------------------
# 3. Boxplot Visualization by Thematic Block
# ------------------------------------------------------------------------------------------------------------

def plot_boxplots(imputed: pd.DataFrame) -> Dict:
    """
    Generate boxplots (z-scored units) for standardized variables grouped 
    by thematic blocks.

    Creates vertical boxplots organized by OECD analytical blocks 
    (INPUT_EXPENDITURE_R&D, OUTPUT_PATENTS, etc.). Outliers are colored 
    by continent for geographical pattern identification.

    Parameters
    ----------
    imputed : pandas.DataFrame
        Preprocessed MSTI dataset with MultiIndex.

    Returns
    -------
    dict
        - figures (matplotlib.figure.Figure): Generated figure
        - axes (list): List of subplot axes
        - metadata (dict): Number of blocks visualized
    """
    print("\n=== STEP : Boxplots for Variables (z-scored) ===")

    # 3.1 --- Apply standard MSTI style ---
    standard_plot_style(grid=True)

    # 3.2 --- Extract numeric columns ---
    numeric = get_numeric_columns(imputed)
    all_vars = numeric.columns.tolist()

    # 3.3 --- Derive continent labels ---
    if "continent" in imputed.columns:
        continent_series = imputed["continent"]
    else:
        continent_series = imputed["zone"].map(COUNTRY_TO_CONTINENT)

    # 3.4 --- Filter thematic blocks to available variables ---
    blocks_available = {
        block_name: [v for v in vars_list if v in all_vars]
        for block_name, vars_list in VARIABLE_BLOCKS.items()
        if any(v in all_vars for v in vars_list)
    }
    n_blocks = len(blocks_available)

    # 3.5 --- Create subplot grid ---
    fig, axes = plt.subplots(
        nrows=n_blocks,
        ncols=1,
        figsize=(18, 4 * n_blocks)
    )

    if n_blocks == 1:
        axes = [axes]

    # 3.6 --- Define boxplot styling ---
    boxprops = dict(facecolor="none", edgecolor="black", linewidth=1.0)
    medianprops = dict(color="red", linewidth=1.2)
    whiskerprops = dict(color="black", linewidth=0.8, linestyle="-")
    capprops = dict(color="black", linewidth=0.8)

    # 3.7 --- Draw boxplots per block ---
    for ax, (block_name, block_vars) in zip(axes, blocks_available.items()):
        block_data = numeric[block_vars]

        # 3.7.1 --- Draw base boxplots ---
        n_vars_in_block = len(block_vars)
        box_width = min(0.6, 8.0 / n_vars_in_block)

        ax.boxplot(
            block_data.values,
            vert=True,
            patch_artist=True,
            widths=box_width,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=dict(marker="o", markersize=0),
        )

        # 3.8 --- Overlay colored outliers by continent ---
        for j, var in enumerate(block_vars):
            series = imputed[[var, "zone"]].copy()
            series["continent"] = continent_series

            # 3.8.1 --- Compute IQR thresholds ---
            q1 = series[var].quantile(0.25)
            q3 = series[var].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 3.8.2 --- Identify outliers ---
            is_outlier = (series[var] < lower_bound) | (series[var] > upper_bound)
            outliers = series[is_outlier]

            # 3.8.3 --- Plot outliers with continent colors ---
            for _, row in outliers.iterrows():
                color = PALETTE_CONTINENT.get(row["continent"], "gray")
                ax.plot(j + 1, row[var], "o", color=color, alpha=0.6, markersize=4)

        # 3.9 --- Configure subplot ---
        ax.set_title(
            block_name.replace("_", " "), 
            fontsize=11, 
            fontweight="semibold"
        )
        ax.set_xticks(range(1, len(block_vars) + 1))
        ax.set_xticklabels(block_vars, rotation=90, ha="right", fontsize=8)
        ax.set_xlim(0.5, max(len(block_vars), 20) + 0.5)
        ax.grid(True, axis="y", which="major")

    # 3.10 --- Apply unified style with legend ---
    legend_handles = [
        plt.Line2D([], [], color=color, marker="o", linestyle="", label=continent)
        for continent, color in PALETTE_CONTINENT.items()
    ]
    
    standard_plot_style(
        fig=fig,
        grid=True,
        legend=True,
        handles=legend_handles,
        legend_title="Continents",
        main_title="Distribution of MSTI Indicators (z-scored - Boxplots)",
        main_title_size=16,
        tight_layout=True
    )

    # 3.11 --- Save figure ---
    output_path = OUTPUT_PATHS["figures"] / "msti_boxplot_variables.png"
    fig.savefig(output_path, bbox_inches="tight")

    return {
        "figure": fig,
        "axes": axes,
        "metadata": {"n_blocks": n_blocks}
    }


# ------------------------------------------------------------------------------------------------------------
# 4. Standalone Module Test
# ------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MSTI UNIVARIATE ANALYSIS - STANDALONE TEST")
    print("=" * 60)

    # 4.1 --- Load imputed dataset ---
    imputed_path = DATA_PATHS["processed"] / "msti_imputed.csv"
    print("\n[1/3] Loading imputed dataset...")
    imputed = pd.read_csv(imputed_path, sep=";")
    print(f"    ✓ Loaded: {imputed.shape}")

    # 4.2 --- Run univariate description ---
    print("\n[2/3] Running statistical description...")
    stats_result = describe_univariate(imputed)
    print(f"    ✓ Variables analyzed: {stats_result['metadata']['n_variables']}")
    print(f"    ✓ Exported to: {stats_result['exports']}")

    # 4.3 --- Run boxplots ---
    print("\n[3/3] Generating boxplots...")
    boxplot_result = plot_boxplots(imputed)
    print(f"    ✓ Blocks visualized: {boxplot_result['metadata']['n_blocks']}")

    print("\n" + "=" * 60)
    print("✓ STANDALONE TEST COMPLETE")
    print("=" * 60)