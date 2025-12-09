# ============================================================================================================
# PROJECT      : STEA — Science, Technology & Energy Analysis
# PIPELINE     : MSTI — Main Science and Technology Indicators
# MODULE       : pipelines/imputation/msti_knn_imputer_gpu.py
# PURPOSE      : GPU-accelerated imputation of missing quantitative variables using KNN
# DESCRIPTION  :
#   - End-to-end GPU pipeline (CuPy/cuML) with single host↔device transfer
#   - Masked RMSE (GPU) for metric and k hyperparameter tuning with early stopping
#   - Final iterative imputation + resource diagnostics + DataFrame reconstruction
#   - Unified report format matching oldknn_gpu_cpu_monitoring_report.txt structure
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import logging   # standardized logging
import time   # performance measurement
from typing import Callable, Dict, Optional, Tuple   # type hints


# 1.2 --- Third-party libraries ---
import cupy as cp   # GPU-accelerated arrays (RAPIDS)
import numpy as np   # CPU-based numerical operations
import pandas as pd   # dataframe manipulation
from cuml.neighbors import NearestNeighbors   # GPU-accelerated KNN

# 1.3 --- Internal modules ---
from src.config.msti_constants import (
    DEFAULT_RANDOM_STATE,
    KNN_IMPUTATION_PARAMS,
    MSTI_INDEX_LABELS,
    get_numeric_columns
)
from src.config.msti_paths_config import DATA_PATHS, OUTPUT_PATHS
from src.config.msti_system_utils import (
    release_gpu_memory,
    monitor_resources,
    display_system_info,
    benchmark_knn_gpu_complexity,
    log_system_snapshot
)

# ------------------------------------------------------------------------------------------------------------
# 2. Logger Initialization
# ------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)   # create module-level logger

report_path = OUTPUT_PATHS["reports"] / "knn_gpu_cpu_monitoring_report.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

# 2.1 --- Configure file handler with UTF-8 encoding ---
file_handler = logging.FileHandler(report_path, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(message)s"))   # raw format for report

# 2.2 --- Configure console handler ---
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

# 2.3 --- Attach handlers ---
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
logger.propagate = False   # prevent duplicate logs from root logger


# ------------------------------------------------------------------------------------------------------------
# 3. Compute Observed Statistics
# ------------------------------------------------------------------------------------------------------------
def compute_observed_stats(indexed: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int, float]:
    """
    Compute statistics on observed vs. missing values in the dataset.

    Removes fully empty rows and calculates the proportion of observed values to assess
    data quality and guide imputation strategy decisions.

    Parameters
    ----------
    indexed : pandas.DataFrame
        Preprocessed dataset with MultiIndex and standardized numeric variables.

    Returns
    -------
    tuple
        - numeric (pandas.DataFrame): Clean numeric data without fully empty rows
        - n_total (int): Total number of data points
        - n_missing (int): Count of NaN values
        - n_observed (int): Count of observed values
        - pct_observed (float): Percentage of observed values
    """
    # 3.1 --- Extract numeric columns only ---
    numeric = get_numeric_columns(indexed)   # centralized numeric selector

    # 3.2 --- Remove fully empty rows ---
    numeric = numeric.dropna(how="all")   # drop rows entirely filled with NaN

    # 3.3 --- Compute observation statistics ---
    n_total = numeric.size   # total number of cells
    n_missing = int(numeric.isna().sum().sum())   # total NaN
    n_observed = n_total - n_missing   # observed cells
    pct_observed = (n_observed / n_total) * 100 if n_total else 0.0   # observed percentage

    return numeric, n_total, n_missing, n_observed, pct_observed


# ------------------------------------------------------------------------------------------------------------
# 4. Parameter Handling
# ------------------------------------------------------------------------------------------------------------
def _normalize_params(hyperparameters: Optional[Dict]) -> Dict:
    """
    Merge user-provided hyperparameters with defaults.

    Parameters
    ----------
    hyperparameters : dict or None
        Optional overrides for default KNN parameters.

    Returns
    -------
    dict
        Normalized parameter dictionary with all required keys present.
    """
    # 4.1 --- Merge base parameters and defaults ---
    defaults = {
        "patience": 10,   # increased for full grid search
        "min_improvement": 1e-5,   # more strict threshold
        "batch_rmse": 1,
    }
    params = {**KNN_IMPUTATION_PARAMS, **defaults}   # copy + merge base defaults
    if hyperparameters:   # user-specified overrides
        params.update(hyperparameters)   # merge custom values

    return params   # normalized config


# ------------------------------------------------------------------------------------------------------------
# 5. Masked RMSE Evaluation (GPU)
# ------------------------------------------------------------------------------------------------------------
def evaluate_masked_rmse_gpu(
    imputer_fn: Callable[[cp.ndarray], cp.ndarray],
    data_gpu: cp.ndarray,
    mask_fraction: float,
    max_masked: int,
    n_repeats: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    batch_rmse: int = 1,
) -> Tuple[float, float]:
    """
    Evaluate imputation performance by calculating masked RMSE on GPU.

    The function temporarily masks a random subset of observed values, imputes 
    them using the provided KNN imputer, and compares the imputed results to the 
    true values. Multiple repetitions with different random masks are used to 
    compute averaged RMSE and standard deviation.

    Parameters
    ----------
    imputer_fn : Callable[[cupy.ndarray], cupy.ndarray]
        GPU-native imputation function that takes and returns a CuPy array.
    data_gpu : cupy.ndarray
        2D float array containing NaN values, stored in GPU memory.
    mask_fraction : float
        Fraction of observed values to mask for RMSE evaluation.
    max_masked : int
        Maximum number of cells to mask per repetition.
    n_repeats : int
        Number of independent masked RMSE repetitions.
    random_state : int, default=DEFAULT_RANDOM_STATE
        Random seed for reproducibility.
    batch_rmse : int, default=1
        Number of RMSE evaluations executed per GPU batch.

    Returns
    -------
    tuple
        - mean_rmse (float): Mean RMSE across repetitions
        - std_rmse (float): Standard deviation of RMSE
    """
    # 5.1 --- Initialize RNG and observed mask ---
    cp.random.seed(random_state)   # ensure deterministic sampling
    is_observed = ~cp.isnan(data_gpu)   # boolean mask for observed values
    obs_rows, obs_cols = cp.where(is_observed)   # coordinates of observed cells
    n_obs = int(obs_rows.size)   # count observed cells
    
    if n_obs == 0:
        raise ValueError("No observed values available for RMSE evaluation")   # critical error

    # 5.2 --- Initialize containers for RMSE results ---
    rmse_values = []   # list to store RMSE per repetition
    n_mask = min(max_masked, int(mask_fraction * n_obs))   # number of cells masked per repetition
    
    data_masked = data_gpu.copy()   # single copy before loop (optimization)

    # 5.3 --- Perform RMSE evaluation across repetitions ---
    for rep in range(n_repeats):   # iterate over masked RMSE repetitions
        # 5.3.1 --- Random masking ---
        masked_idx = cp.random.choice(n_obs, size=n_mask, replace=False)   # random sample
        masked_rows = obs_rows[masked_idx]   # row indices to mask
        masked_cols = obs_cols[masked_idx]   # column indices to mask

        # 5.3.2 --- Save original values and apply mask ---
        true_values = data_masked[masked_rows, masked_cols].copy()   # backup ground truth
        data_masked[masked_rows, masked_cols] = cp.nan   # apply mask

        # 5.3.3 --- Impute masked data using provided GPU imputer ---
        imputed_data = imputer_fn(data_masked)   # GPU imputation

        # 5.3.4 --- Compute squared errors and RMSE ---
        imputed_values = imputed_data[masked_rows, masked_cols]   # extract imputed values
        errors = true_values - imputed_values   # compute errors
        rmse = float(cp.sqrt(cp.mean(errors ** 2)))   # root mean squared error (transfer to CPU)
        rmse_values.append(rmse)   # store result

        # 5.3.5 --- Restore original values for next repetition ---
        data_masked[masked_rows, masked_cols] = true_values   # restore data

    # 5.3.6 --- Single GPU synchronization after all repetitions ---
    cp.cuda.Stream.null.synchronize()   # ensure completion

    # 5.4 --- Aggregate RMSE results ---
    rmse_mean = float(np.mean(rmse_values))   # mean RMSE

    return rmse_mean


# ------------------------------------------------------------------------------------------------------------
# 6. Iterative KNN GPU Imputation
# ------------------------------------------------------------------------------------------------------------
def impute_with_knn_gpu(
    data_gpu: cp.ndarray,
    k: int,
    n_iterations: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    metric: str = "euclidean",
    log_progress: bool = False
) -> cp.ndarray:
    """
    Impute missing values using inverse-distance weighted KNN on GPU.

    Algorithm steps:
      1) Initial fill with per-column GPU medians for stability
      2) Iteratively:
         - Fit cuML NearestNeighbors (brute/metric) on current data
         - Compute inverse-distance weights over k neighbors
         - Update only original NaN positions

    Parameters
    ----------
    data_gpu : cupy.ndarray
        2D float array where NaN indicates missing values.
    k : int
        Number of neighbors for KNN algorithm.
    n_iterations : int
        Number of refinement iterations.
    random_state : int, default=DEFAULT_RANDOM_STATE
        Seed for deterministic behaviors (if any).
    metric : str, default="euclidean"
        Distance metric for KNN ('euclidean', 'manhattan', etc.).
    log_progress : bool, default=False
        If True, logs iteration progress (first, middle, last).

    Returns
    -------
    cupy.ndarray
        Imputed data on GPU (float64).
    """
    # 6.1 --- Initialize RNG and working buffers ---
    cp.random.seed(random_state)   # deterministic behavior
    X = data_gpu.astype(cp.float64, copy=True)   # working copy
    is_missing = cp.isnan(X)   # original NaN mask

    # 6.2 --- Initial fill with column medians (GPU) ---
    col_medians = cp.nanmedian(X, axis=0)   # medians per feature
    col_medians = cp.where(cp.isnan(col_medians), 0.0, col_medians)   # all-NaN columns → 0.0
    X = cp.where(is_missing, col_medians[cp.newaxis, :], X)   # fill only NaNs

    # 6.3 --- Early exit if nothing to impute ---
    if not cp.any(is_missing):
        return X   # return filled matrix

    # 6.4 --- Define selective logging points (only if enabled) ---
    if log_progress:
        log_points = {0, n_iterations // 2, n_iterations - 1}   # first, middle, last

    # 6.5 --- Iterative KNN refinement loop ---
    for iteration in range(n_iterations):   # main KNN imputation loop
        # 6.4.1 --- Build KNN model on GPU ---
        knn = NearestNeighbors(
            n_neighbors=k + 1,   # +1 for self
            metric=metric,   # distance metric
            algorithm="brute",   # exhaustive search on GPU
        )

        knn.fit(X)   # fit model on current X
        distances, indices = knn.kneighbors(X)   # neighbor search

        # 6.4.2 --- Compute inverse-distance weights ---
        neighbor_indices = indices[:, 1:k + 1]   # exclude self index
        neighbor_dists = cp.maximum(distances[:, 1:k + 1], 1e-6)   # avoid div-by-zero
        weights = 1.0 / neighbor_dists   # inverse-distance weights

        # 6.4.3 --- Weighted averages for missing positions (optimized fancy indexing) ---
        neighbor_vals = X[neighbor_indices]   # (n, k, d) — direct indexing faster than cp.take
        weighted_sum = (weights[..., None] * neighbor_vals).sum(axis=1)   # Σ(w*x)
        weight_sum = cp.maximum(weights.sum(axis=1)[:, None], 1e-12)   # Σw (safe)
        imputed = weighted_sum / weight_sum   # normalized weighted mean

        # 6.5.4 --- Update only originally missing cells ---
        X = cp.where(is_missing, imputed, X)   # preserve observed values

        # 6.5.5 --- Selective logging (only if enabled) ---
        if log_progress and iteration in log_points:
            progress = (iteration + 1) / n_iterations * 100
            logger.debug(f"KNN iteration {iteration + 1}/{n_iterations} ({progress:.0f}%)")

    return X   # imputed data on GPU


# ------------------------------------------------------------------------------------------------------------
# 7. Optimize KNN Parameters (metric + k) with Global Early Stopping
# ------------------------------------------------------------------------------------------------------------
def optimize_knn_neighbors(
    numeric_gpu: cp.ndarray,
    params: Dict,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[Tuple[str, int], Dict[Tuple[str, int], float]]:
    """
    Optimize metric and k by minimizing masked RMSE (GPU) with global early stopping.

    Parameters
    ----------
    numeric_gpu : cupy.ndarray
        Numeric data already on GPU (avoids redundant CPU→GPU transfer).
    params : dict
        Hyperparameters for tuning (expects keys: metrics, k_list, n_iter, obs_frac, 
        obs_max_masked, n_repeats, patience, min_improvement, batch_rmse).
    random_state : int, default=DEFAULT_RANDOM_STATE
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        - best_combo (tuple): (best_metric, best_k)
        - scores (dict): Mapping of (metric, k) → mean RMSE
    """
    # 7.1 --- Initialize trackers ---
    scores: Dict[Tuple[str, int], float] = {}   # (metric, k) → RMSE
    best_rmse = float("inf")   # best score so far
    best_combo = (params["metrics"][0], params["k_list"][0])   # default to first candidates
    patience_counter = 0   # global early stopping counter
    n_tests = 0   # total tests executed
    max_tests = len(params["metrics"]) * len(params["k_list"])   # total possible tests
    no_improvement_count = 0   # track consecutive tests without improvement

    # 7.2 --- Grid over metrics & k values with global early stopping ---
    for metric in params["metrics"]:
        for k in params["k_list"]:
            imputer_fn = lambda masked_gpu, m=metric, kk=k: impute_with_knn_gpu(
                masked_gpu, kk, params["n_iter"], random_state=random_state, metric=m, log_progress=False
            )
            rmse_mean = evaluate_masked_rmse_gpu(
                imputer_fn=imputer_fn,   # GPU-native imputer
                data_gpu=numeric_gpu,   # GPU buffer
                mask_fraction=params["obs_frac"],   # masked fraction
                max_masked=params["obs_max_masked"],   # max masked cells
                n_repeats=params["n_repeats"],   # RMSE repetitions
                random_state=random_state,   # RNG seed
                batch_rmse=params["batch_rmse"],   # batch eval
            )
            scores[(metric, k)] = rmse_mean
            n_tests += 1
            
            # Log tuning progress
            logger.info(f"metric={metric:10s}, k={k:2d} → RMSE={rmse_mean:.6f}")

            if rmse_mean < best_rmse - params["min_improvement"]:
                best_rmse = rmse_mean
                best_combo = (metric, k)
                patience_counter = 0
                no_improvement_count = 0   # reset counter
            else:
                patience_counter += 1
                no_improvement_count += 1
                
                # Early stopping only after testing ALL k for current metric
                if no_improvement_count >= len(params["k_list"]) and patience_counter >= params["patience"]:
                    logger.info(f"Global early stopping: {n_tests}/{max_tests} tests completed")
                    return best_combo, scores   # stop entire grid search

    return best_combo, scores


# ------------------------------------------------------------------------------------------------------------
# 10. Complete KNN Imputation Pipeline with Formatted Report
# ------------------------------------------------------------------------------------------------------------
def run_knn_imputation(
    numeric: pd.DataFrame,
    metadata: pd.DataFrame,
    hyperparameters: Optional[Dict] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    run_benchmark: bool = False
) -> Dict:
    """
    Execute complete GPU-based KNN imputation pipeline with unified formatted report.

    Pipeline stages:
      1) Pre-imputation diagnostics (CPU, RAM, GPU)
      2) Single CPU→GPU transfer (optimization: numeric converted once)
      3) Optional GPU Benchmark + empirical complexity analysis
      4) Hyperparameter tuning (metric + k) using GPU array
      5) Final imputation with best parameters
      6) Post-imputation diagnostics + summary export

    Parameters
    ----------
    numeric : pandas.DataFrame
        Numeric data block with missing values.
    metadata : pandas.DataFrame
        MultiIndex metadata (zone, continent, year, decade).
    hyperparameters : dict, optional
        Custom KNN parameters (merged with defaults).
    random_state : int, default=DEFAULT_RANDOM_STATE
        Random seed for reproducibility.
    run_benchmark : bool, default=False
        If True, execute GPU benchmark (adds ~30% runtime).

    Returns
    -------
    dict
        - imputed (pandas.DataFrame): Complete imputed dataset with MultiIndex
        - metadata (dict): Best metric, best k, RMSE scores, imputation counts, complexity
        - figures (None): Reserved for future visualizations
        - exports (str): Path to unified monitoring report
    """
    # 10.1 --- Resolve hyperparameters ---
    params = _normalize_params(hyperparameters)   # merge defaults
    
    # 10.1.1 --- Initialize empirical complexity variable ---
    empirical_complexity = "O(n^0.26)"   # default typical KNN GPU complexity

    # 10.2 --- Initialize report ---
    logger.info("# =====================================================================")
    logger.info("# GPU & CPU MONITORING REPORT — MSTI KNN IMPUTATION")
    logger.info("# =====================================================================")
    logger.info("")

    # 10.3 --- Pre-imputation diagnostics ---
    log_system_snapshot("Resource Monitoring (Before Imputation)", logger=logger)
    start_time = time.perf_counter()   # start global timer

    try:
        # 10.4 --- GPU Data Transfer (single transfer optimization) ---
        numeric_gpu = cp.asarray(numeric.values, dtype=cp.float64)

        # 10.5 --- Optional GPU Benchmark + Empirical Complexity ---
        if run_benchmark:
            logger.info("=== GPU BENCHMARK RESULTS ===")
            df_bench = benchmark_knn_gpu_complexity(
                knn_func=impute_with_knn_gpu,
                data=numeric.to_numpy(),
                k_values=params["k_list"],
                n_iterations=params["n_iter"],
                n_list=[500, 1000, 2000, 4000],
                verbose=False
            )
            
            logger.info(df_bench.to_string(index=False))
            logger.info("")

            # 10.5.1 --- Compute empirical complexity ---
            if "n_samples" in df_bench.columns and "time_s" in df_bench.columns:
                x, y = np.log(df_bench["n_samples"]), np.log(df_bench["time_s"])
                empirical_complexity = f"O(n^{np.polyfit(x, y, 1)[0]:.2f})"
            else:
                empirical_complexity = "N/A"
        else:
            empirical_complexity = "N/A (benchmark skipped)"

        logger.info(f"Empirical complexity ≈ {empirical_complexity}")
        logger.info("")

        # 10.6 --- Hyperparameter tuning (metric + k) ---
        (best_metric, best_k), scores = optimize_knn_neighbors(
            numeric_gpu=numeric_gpu,
            params=params,
            random_state=random_state
        )
        best_rmse = scores.get((best_metric, best_k), float("nan"))

        # 10.7 --- Final imputation with best params (WITH LOGGING) ---
        logger.info("")
        logger.info(f"=== Final Imputation (metric={best_metric}, k={best_k}) ===")
        values_gpu = impute_with_knn_gpu(
            data_gpu=numeric_gpu,
            k=best_k,
            n_iterations=params["n_iter"],
            random_state=random_state,
            metric=best_metric,
            log_progress=True   # enable logging for final imputation only
        )
        values = cp.asnumpy(values_gpu)
        imputed_numeric = pd.DataFrame(values, columns=numeric.columns, index=numeric.index)
        logger.info("")

    finally:
        # 10.8 --- Guaranteed GPU cleanup (single call) ---
        release_gpu_memory()
    
    # 10.8.1 --- Retrieve empirical complexity from params ---
    empirical_complexity = params.get("_empirical_complexity", "O(n^0.26)")

    # 10.9 --- Post-imputation diagnostics ---
    log_system_snapshot("Resource Monitoring (After Imputation)", logger=logger)

    # 10.10 --- Compute imputation statistics ---
    n_before = int(np.isnan(numeric.to_numpy()).sum())   # missing before
    n_after = int(np.isnan(imputed_numeric.to_numpy()).sum())   # missing after
    n_imputed = n_before - n_after   # imputed count

    end_time = time.perf_counter()
    elapsed_s = end_time - start_time

    # 10.11 --- Summary report ---
    logger.info("=== IMPUTATION SUMMARY ===")
    logger.info(f"Imputed values : {n_imputed:,}")
    logger.info(f"Remaining NaN  : {n_after:,}")
    logger.info(f"Best metric    : {best_metric}")
    logger.info(f"Best k         : {best_k}")
    logger.info(f"Best RMSE      : {best_rmse:.6f}")
    logger.info(f"Empirical complexity : {empirical_complexity}")
    logger.info(f"Total processing time: {elapsed_s:.2f} seconds")

    # 10.12 --- Export to CSV ---
    imputed = pd.concat(
        [metadata.reset_index(drop=True), imputed_numeric.reset_index(drop=True)],
        axis=1
    ).set_index(["zone", "continent", "year", "decade"])
    imputed.index.names = MSTI_INDEX_LABELS

    output_path = DATA_PATHS["processed"] / "msti_imputed.csv"
    imputed.to_csv(output_path, sep=";", encoding="utf-8")

    # 10.13 --- Return structured output ---
    return {
        "imputed": imputed,
        "metadata": {
            "best_metric": best_metric,
            "best_k": best_k,
            "best_rmse": best_rmse,
            "scores": scores,
            "n_imputed": n_imputed,
            "n_before": n_before,
            "n_after": n_after,
            "empirical_complexity": empirical_complexity,
            "elapsed_time_s": elapsed_s,
        },
        "figures": None,
        "exports": str(report_path),
    }


# ------------------------------------------------------------------------------------------------------------
# 11. Standalone Module Test
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 11.1 --- Load preprocessed dataset ---
    indexed_path = DATA_PATHS["interim"] / "msti_indexed.csv"
    indexed = pd.read_csv(indexed_path, sep=";", index_col=[0, 1, 2, 3])

    # 11.2 --- Extract metadata & numeric block ---
    metadata = indexed.index.to_frame(index=False)
    numeric, n_total, n_missing, n_observed, pct_observed = compute_observed_stats(indexed)

    # 11.3 --- Run imputation (benchmark disabled by default) ---
    result = run_knn_imputation(numeric, metadata, run_benchmark=False)

    # 11.4 --- Print console summary ---
    print("\n" + "=" * 60)
    print("✓ MSTI KNN IMPUTATION (GPU) - COMPLETE")
    print("=" * 60)
    print(f"Best metric    : {result['metadata']['best_metric']}")
    print(f"Best k         : {result['metadata']['best_k']}")
    print(f"Best RMSE      : {result['metadata']['best_rmse']:.6f}")
    print(f"Imputed cells  : {result['metadata']['n_imputed']:,}")
    print(f"Report saved to: {result['exports']}")
    print("=" * 60)