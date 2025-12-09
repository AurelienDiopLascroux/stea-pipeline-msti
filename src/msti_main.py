# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : msti_main.py
# PURPOSE      : Main orchestrator for MSTI pipeline execution
# DESCRIPTION  :
#   - Orchestrates the complete MSTI analysis pipeline from ingestion to visualization.
#   - Executes: data loading → indexing → imputation → topological projection → 
#     descriptive statistics → multivariate analysis.
#   - Provides execution tracking, error handling, and performance monitoring.
#   - Includes optimizations: GPU cleanup, parallel MFA plotting, progress bars, metadata export.
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import argparse   # command-line argument parsing
import gc   # garbage collection
import json   # metadata export
import logging   # logging configuration
import sys   # system operations
import time   # performance measurement
from concurrent.futures import ThreadPoolExecutor, as_completed   # parallel execution
from pathlib import Path   # path management

# 1.2 --- Third-party libraries ---
import numpy as np   # numerical operations
import pandas as pd   # dataframe manipulation
from tqdm import tqdm   # progress bars

# 1.3 --- Internal modules ---
from src.s05_analysis.msti_analysis_univariate import (
    describe_univariate,
    plot_boxplots
)
from src.s05_analysis.msti_cah_mfa import (
    cluster_variables_hierarchical,
    plot_cattell_criterion,
    plot_mfa_correlation_circle_f12,
    plot_mfa_correlation_circle_f13,
    plot_mfa_projection_continents_f12,
    plot_mfa_projection_continents_f13,
    plot_mfa_projection_countries_f12,
    plot_mfa_projection_countries_f13,
    run_mfa_projection
)
from src.s05_analysis.msti_corr_analysis import plot_correlation_matrix
from src.s03_imputation.msti_knn_imputer_gpu import (
    compute_observed_stats,
    run_knn_imputation
)
from src.s02_indexing.msti_indexing import (
    add_dimensions,
    build_indicators,
    reshape_wide,
    select_core_columns,
    set_index,
    standardize
)
from src.s01_ingestion.msti_ingestion_load_data import load_raw_data
from src.config.msti_constants import DEFAULT_RANDOM_STATE, MSTI_INDEX_LABELS
from src.config.msti_paths_config import DATA_PATHS, OUTPUT_PATHS
from src.config.msti_system_utils import display_system_info
from src.config.msti_variables_mapping import COUNTRY_TO_CONTINENT
from src.s04_visualization.msti_umap_projection import (
    project_umap3d_clusters,
    project_umap3d_observations
)


# ------------------------------------------------------------------------------------------------------------
# 2. Project Configuration
# ------------------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------------------
# 3. Logging Configuration
# ------------------------------------------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for MSTI pipeline execution.
    
    Sets up console handler with simple format for ETL modules.
    All child loggers inherit this configuration.
    
    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # 3.1 --- Remove existing handlers ---
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 3.2 --- Configure root logger ---
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
    
    # 3.3 --- Ensure pipelines package loggers are visible ---
    logging.getLogger("pipelines").setLevel(level)


# ------------------------------------------------------------------------------------------------------------
# 4. Environment Validation
# ------------------------------------------------------------------------------------------------------------
def validate_environment() -> None:
    """
    Validate pipeline prerequisites before execution.
    
    Checks for:
    - RAPIDS GPU libraries availability
    - Input data file existence
    - Output directory structure
    
    Raises
    ------
    EnvironmentError
        If critical prerequisites are missing.
    """
    # 4.1 --- Check GPU libraries ---
    errors = []
    
    try:
        import cudf   # test RAPIDS import
        logger.debug("✓ RAPIDS cuDF available")
    except ImportError:
        errors.append("❌ RAPIDS not installed (GPU acceleration unavailable)")
    
    # 4.2 --- Check input data ---
    raw_path = DATA_PATHS["raw"] / "msti_raw.csv"
    if not raw_path.exists():
        errors.append(f"❌ Input file missing: {raw_path}")
    else:
        logger.debug(f"✓ Input file found: {raw_path}")
    
    # 4.3 --- Ensure output directories exist ---
    for key in ["figures", "reports"]:
        path = OUTPUT_PATHS[key]
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {path}")
        else:
            logger.debug(f"✓ Directory exists: {path}")
    
    # 4.4 --- Raise if critical errors ---
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"\n{error_msg}")
        raise EnvironmentError("Pipeline prerequisites not met")


# ------------------------------------------------------------------------------------------------------------
# 5. GPU Memory Cleanup
# ------------------------------------------------------------------------------------------------------------
def cleanup_gpu_memory() -> None:
    """
    Release GPU memory after intensive GPU operations.
    
    Frees all blocks from CuPy default and pinned memory pools.
    Safe to call even if CuPy is not available.
    """
    try:
        import cupy as cp   # import CuPy
        cp.get_default_memory_pool().free_all_blocks()   # free default pool
        cp.get_default_pinned_memory_pool().free_all_blocks()   # free pinned pool
        logger.debug("✓ GPU memory released")
    except ImportError:
        logger.debug("CuPy not available, skipping GPU cleanup")


# ------------------------------------------------------------------------------------------------------------
# 6. Main Pipeline Execution
# ------------------------------------------------------------------------------------------------------------
def main(verbose: bool = False) -> None:
    """
    Execute complete MSTI pipeline from ingestion to multivariate analysis.
    
    Pipeline stages:
    1. Data Ingestion: Load raw OECD MSTI dataset
    2. Indexing & Standardization: Clean, reshape, z-score normalization, MultiIndex
    3. Imputation: GPU-accelerated KNN with inverse-distance weighting
    4. Topological Projection: UMAP 3D with vectorized repulsion forces
    5. Descriptive Statistics: Univariate distributions and boxplots
    6. Multivariate Analysis: Pearson correlation, Ward-linkage CAH, weighted MFA
    
    Each stage logs progress and exports intermediate results.
    
    Parameters
    ----------
    verbose : bool, default=False
        Enable detailed DEBUG-level logging.
    """
    # 6.1 --- Configure logging ---
    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=level)
    
    # 6.2 --- Validate environment ---
    try:
        validate_environment()
    except EnvironmentError as e:
        logger.error(f"Environment validation failed: {e}")
        sys.exit(1)
    
    # 6.3 --- Display system info ---
    display_system_info()
    
    # 6.4 --- Pipeline header ---
    print("\n" + "="*70)
    print("MSTI PIPELINE EXECUTION")
    print("="*70)
    
    start_time = time.time()
    pipeline_metadata = {}   # store execution metadata
    
    try:
        # ========================================
        # STAGE 1: DATA INGESTION
        # ========================================
        print("\n[STAGE 1/6] DATA INGESTION")
        print("-" * 70)
        
        stage_start = time.time()
        logger.info("Loading raw OECD MSTI dataset...")
        raw = load_raw_data("msti_raw.csv")
        logger.info(f"✓ Raw dataset loaded: {raw.shape[0]:,} rows × {raw.shape[1]} cols")
        pipeline_metadata["stage1_time"] = time.time() - stage_start
        pipeline_metadata["raw_shape"] = raw.shape
        
        # ========================================
        # STAGE 2: INDEXING & STANDARDIZATION
        # ========================================
        print("\n[STAGE 2/6] INDEXING & STANDARDIZATION")
        print("  • Indicator mapping (measure × unit)")
        print("  • Spatio-temporal dimensions (continent, decade)")
        print("  • Wide format reshaping")
        print("  • Z-score normalization (μ=0, σ=1)")
        print("  • MultiIndex structuring [zone, continent, year, decade]")
        print("-" * 70)
        
        stage_start = time.time()
        logger.info("Step 1/5: Selecting core columns...")
        mapping_file = DATA_PATHS["mappings"] / "msti_indicator_mapping.json"
        
        indexed = (
            raw
            .pipe(select_core_columns)
            .pipe(build_indicators, mapping_file=mapping_file)
            .pipe(add_dimensions, geo_mapping=COUNTRY_TO_CONTINENT)
            .pipe(reshape_wide)
            .pipe(standardize)
            .pipe(set_index)
        )
        
        logger.info(f"✓ Indexing complete: {indexed.shape[0]:,} observations × {indexed.shape[1]} variables")
        pipeline_metadata["stage2_time"] = time.time() - stage_start
        pipeline_metadata["indexed_shape"] = indexed.shape
        
        # 6.4.1 --- Memory cleanup ---
        del raw   # free raw data memory
        gc.collect()
        logger.debug("✓ Raw data memory released")
        
        # ========================================
        # STAGE 3: GPU-ACCELERATED IMPUTATION
        # ========================================
        print("\n[STAGE 3/6] GPU-ACCELERATED IMPUTATION (cuML)")
        print("  • Algorithm: Weighted KNN (inverse-distance kernel)")
        print("  • Hyperparameter tuning: k ∈ {2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50}")
        print("  • Validation: Masked RMSE with early stopping")
        print("  • Hardware: RAPIDS cuDF + CuPy + cuML")
        print("-" * 70)
        
        stage_start = time.time()
        logger.info("Computing observed statistics...")
        metadata = indexed.index.to_frame(index=False)
        numeric, total, missing, observed, pct = compute_observed_stats(indexed)
        logger.info(f"✓ Data quality: {pct:.2f}% observed ({observed:,}/{total:,} cells)")
        
        logger.info("Running KNN imputation (GPU)...")
        result = run_knn_imputation(numeric, metadata)
        imputed = result["imputed"]
        if not isinstance(imputed.index, pd.MultiIndex):
            if all(col in imputed.columns for col in ["zone", "continent", "year", "decade"]):
                imputed = imputed.set_index(["zone", "continent", "year", "decade"])
                imputed.index.names = MSTI_INDEX_LABELS

        logger.info(
            f"✓ Imputation complete: {result['metadata']['n_imputed']:,} values imputed | "
            f"Best k={result['metadata']['best_k']} | "
            f"RMSE={result['metadata']['best_rmse']:.6f}"
        )
        
        pipeline_metadata["stage3_time"] = time.time() - stage_start
        pipeline_metadata["imputation"] = {
            "n_imputed": result['metadata']['n_imputed'],
            "best_k": result['metadata']['best_k'],
            "best_rmse": result['metadata']['best_rmse'],
            "observed_pct": pct
        }
        
        # 6.4.2 --- GPU memory cleanup ---
        cleanup_gpu_memory()
        
        # 6.4.3 --- Memory cleanup ---
        del numeric, metadata, indexed   # free intermediate data
        gc.collect()
        logger.debug("✓ Intermediate data memory released")
        
        # ========================================
        # STAGE 4: TOPOLOGICAL PROJECTION
        # ========================================
        print("\n[STAGE 4/6] TOPOLOGICAL PROJECTION (UMAP 3D)")
        print("  • Algorithm: Uniform Manifold Approximation and Projection")
        print("  • Parameters: n_neighbors=15, min_dist=0.1, metric=euclidean")
        print("  • Post-processing: Vectorized repulsion (cKDTree + float32)")
        print("  • Clustering: HDBSCAN (min_cluster_size=15, min_samples=5)")
        print("-" * 70)
        
        stage_start = time.time()
        
        # 6.4.4 --- UMAP observations ---
        print("\n[4.1] UMAP 3D projection (observations)...")
        fig_obs_plotly, fig_obs_mpl = project_umap3d_observations(
            imputed,
            apply_repulsion=True,
            random_state=DEFAULT_RANDOM_STATE
        )
        print("✓ Observations projection complete (static + interactive)")
        
        # 6.4.5 --- UMAP clusters ---
        print("\n[4.2] UMAP 3D projection (clusters)...")
        fig_clust_plotly, fig_clust_mpl = project_umap3d_clusters(
            imputed,
            min_cluster_size=15,
            min_samples=5,
            apply_repulsion=True,
            random_state=DEFAULT_RANDOM_STATE
        )
        print("✓ Cluster projection complete (static + interactive)")
        
        pipeline_metadata["stage4_time"] = time.time() - stage_start
        
        # ========================================
        # STAGE 5: DESCRIPTIVE STATISTICS
        # ========================================
        print("\n[STAGE 5/6] DESCRIPTIVE STATISTICS (UNIVARIATE)")
        print("  • Central tendency: mean, median")
        print("  • Dispersion: std, IQR, range")
        print("  • Distribution: quartiles, boxplots by thematic blocks")
        print("-" * 70)
        
        stage_start = time.time()
        
        # 6.4.6 --- Univariate analysis ---
        print("\n[5.1] Univariate descriptive statistics...")
        stats_result = describe_univariate(imputed)
        print(f"✓ Statistics computed for {stats_result['metadata']['n_variables']} variables")
        print(f"  Exported to: {stats_result['exports']}")
        
        # 6.4.7 --- Boxplots ---
        print("\n[5.2] Generating boxplots...")
        print("DEBUG — columns in imputed:", imputed.columns.tolist())
        print("DEBUG — head:")
        print(imputed.head())

        imputed_for_boxplots = imputed.reset_index()
        boxplot_result = plot_boxplots(imputed_for_boxplots)
        print(f"✓ Boxplots generated for {boxplot_result['metadata']['n_blocks']} thematic blocks")
        
        pipeline_metadata["stage5_time"] = time.time() - stage_start
        
        # ========================================
        # STAGE 6: MULTIVARIATE ANALYSIS
        # ========================================
        print("\n[STAGE 6/6] MULTIVARIATE ANALYSIS")
        print("  • Correlation: Pearson r coefficient matrix (89×89)")
        print("  • CAH: Ward linkage + Euclidean distance")
        print("  • MFA: Block-weighted PCA (w_g = 1/√λ₁) + spline barycenters")
        print("-" * 70)
        
        stage_start = time.time()
        
        # 6.4.8 --- Correlation analysis ---
        print("\n[6.1] Correlation matrix (Pearson)...")
        corr_result = plot_correlation_matrix(imputed, mask_upper=True)
        print(f"✓ Correlation matrix computed: {corr_result['metadata']['shape']}")
        print(f"  Mean |correlation|: {corr_result['metadata']['mean_abs_correlation']:.3f}")
        print(f"  Exported to: {corr_result['exports']}")
        
        # 6.4.9 --- Hierarchical clustering ---
        print("\n[6.2] Hierarchical clustering (Ward-linkage CAH)...")
        cah_result = cluster_variables_hierarchical(imputed)
        print(f"✓ CAH complete:")
        print(f"  Elbow method: {cah_result['metadata']['n_clusters_elbow']} clusters")
        print(f"  Max silhouette: {cah_result['metadata']['n_clusters_silhouette']} clusters")
        print(f"  Best silhouette score: {cah_result['metadata']['best_silhouette']:.3f}")
        print(f"  Dendrogram: {cah_result['figures']['dendrogram']}")
        print(f"  Evaluation: {cah_result['figures']['evaluation']}")
        
        pipeline_metadata["cah"] = {
            "n_clusters_elbow": cah_result['metadata']['n_clusters_elbow'],
            "n_clusters_silhouette": cah_result['metadata']['n_clusters_silhouette'],
            "best_silhouette": cah_result['metadata']['best_silhouette']
        }
        
        # 6.4.10 --- Multiple factor analysis ---
        print("\n[6.3] Multiple factor analysis (MFA)...")
        print("  • Stage 1: Block-wise PCA with adaptive weighting")
        print("  • Stage 2: Global PCA on weighted sparse block axes")
        print("  • Metrics: COS² (quality), CTR (contributions)")
        mfa_exports, mfa_figures = run_mfa_projection(imputed)
        from src.s05_analysis.msti_cah_mfa import (
            obs_coords_df, 
            explained_variance, 
            clean_data, 
            correlations_df
        )
        print(f"✓ MFA complete:")
        print(f"  Inertia exported: {mfa_exports['inertia']}")
        print(f"  Variable metrics: {mfa_exports['variable_metrics']}")
        print(f"  Observation metrics: {mfa_exports['obs_metrics_excel']}")
        print(f"    • Sparse PCA dataset → {mfa_exports['mfa_final_csv']}")

        pipeline_metadata["mfa"] = {
            "eigenvalues": mfa_exports["eigenvalues"][:3],
            "explained_variance": mfa_exports["explained_variance"][:3]
        }
        
        # 6.4.11 --- MFA Visualizations (PARALLEL) ---
        print("\n[6.4] Generating MFA visualizations (parallel processing)...")
        print("  • Cattell criterion (scree plot)")
        print("  • Correlation circles (variable-component correlations)")
        print("  • Spline-based temporal barycenters (continent trajectories)")
        print("  • Country projections with spline smoothing (s=2.0)")
        
        # Define visualization tasks
        spline_smoothing = 2.0
        viz_tasks = [
            ("Cattell scree plot", lambda: plot_cattell_criterion(
                np.array(mfa_exports["eigenvalues"]), explained_variance
            )),
            ("Correlation circle (F1×F2)", lambda: plot_mfa_correlation_circle_f12(
                correlations_df, explained_variance
            )),
            ("Correlation circle (F1×F3)", lambda: plot_mfa_correlation_circle_f13(
                correlations_df, explained_variance
            )),
            ("Continent projection (F1×F2)", lambda: plot_mfa_projection_continents_f12(
                obs_coords_df, explained_variance, clean_data
            )),
            ("Continent projection (F1×F3)", lambda: plot_mfa_projection_continents_f13(
                obs_coords_df, explained_variance, clean_data
            )),
            ("Country projection (F1×F2)", lambda: plot_mfa_projection_countries_f12(
                obs_coords_df, explained_variance, clean_data, spline_s=spline_smoothing
            )),
            ("Country projection (F1×F3)", lambda: plot_mfa_projection_countries_f13(
                obs_coords_df, explained_variance, clean_data, spline_s=spline_smoothing
            )),
        ]
        
        # Execute visualizations in parallel with progress bar
        mfa_viz_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_name = {executor.submit(task): name for name, task in viz_tasks}
            
            with tqdm(total=len(viz_tasks), desc="MFA visualizations", unit="plot") as pbar:
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        fig, path = future.result()
                        mfa_viz_results[name] = path
                        print(f"  ✓ {name}: {path}")
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"  ✗ {name} failed: {e}")
                        pbar.update(1)
        
        pipeline_metadata["stage6_time"] = time.time() - stage_start
        pipeline_metadata["mfa_visualizations"] = len(mfa_viz_results)
        
        # ========================================
        # PIPELINE COMPLETE
        # ========================================
        elapsed_time = time.time() - start_time
        pipeline_metadata["total_time_seconds"] = elapsed_time
        pipeline_metadata["execution_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pipeline_metadata["random_state"] = DEFAULT_RANDOM_STATE
        pipeline_metadata["spline_smoothing"] = spline_smoothing
        
        # 6.4.12 --- Export pipeline metadata ---
        metadata_path = OUTPUT_PATHS["reports"] / "pipeline_metadata.json"
        with open(metadata_path, "w") as f:
            # Convertir les types numpy en types Python natifs pour la sérialisation JSON
            json.dump(pipeline_metadata, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
        logger.info(f"✓ Pipeline metadata exported: {metadata_path}")
        
        # 6.4.13 --- Final summary ---
        print("\n" + "="*70)
        print("✓ MSTI PIPELINE COMPLETE")
        print("="*70)
        print(f"Total execution time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
        print(f"\nOutputs saved to:")
        print(f"  • Figures: {OUTPUT_PATHS['figures']}")
        print(f"  • Reports: {OUTPUT_PATHS['reports']}")
        print(f"  • Data: {DATA_PATHS['processed']}")
        print(f"  • Data (Sparse PCA): {DATA_PATHS['processed']}")
        print(f"  • Metadata: {metadata_path}")
        print("\nGenerated visualizations:")
        print(f"  [Topological]")
        print(f"    • UMAP 3D observations (with repulsion forces)")
        print(f"    • UMAP 3D clusters (HDBSCAN)")
        print(f"  [Univariate]")
        print(f"    • Boxplots by thematic blocks ({boxplot_result['metadata']['n_blocks']} blocks)")
        print(f"  [Multivariate - Correlation]")
        print(f"    • Pearson correlation matrix (89×89)")
        print(f"  [Multivariate - CAH]")
        print(f"    • Ward-linkage dendrogram")
        print(f"    • Elbow + Silhouette evaluation plots")
        print(f"  [Multivariate - MFA]")
        print(f"    • Cattell scree plot (eigenvalues)")
        print(f"  [Multivariate - MFA (Sparse PCA)]")
        print(f"    • Correlation circles (F1×F2, F1×F3)")
        print(f"    • Continent projections (spline barycenters)")
        print(f"    • Country projections (spline-smoothed trajectories)")
        print("\nPerformance summary:")
        print(f"  • Stage 1 (Ingestion): {pipeline_metadata['stage1_time']:.1f}s")
        print(f"  • Stage 2 (Indexing): {pipeline_metadata['stage2_time']:.1f}s")
        print(f"  • Stage 3 (Imputation): {pipeline_metadata['stage3_time']:.1f}s")
        print(f"  • Stage 4 (UMAP): {pipeline_metadata['stage4_time']:.1f}s")
        print(f"  • Stage 5 (Descriptive): {pipeline_metadata['stage5_time']:.1f}s")
        print(f"  • Stage 6 (Multivariate): {pipeline_metadata['stage6_time']:.1f}s")
        print("="*70 + "\n")
        
    except Exception as e:
        # 6.5 --- Error handling ---
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}", exc_info=True)
        print("\n" + "="*70)
        print("❌ PIPELINE FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        print("Check logs for detailed traceback.")
        print("="*70 + "\n")
        raise


# ------------------------------------------------------------------------------------------------------------
# 7. Entry Point
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 7.1 --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="MSTI Pipeline - Main Science and Technology Indicators Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python msti_main.py                 # Run with default settings
  python msti_main.py --verbose       # Enable detailed logging
  python msti_main.py -v              # Short form for verbose
        """
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    args = parser.parse_args()
    
    # 7.2 --- Execute pipeline ---
    main(verbose=args.verbose)