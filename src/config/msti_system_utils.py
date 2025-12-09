# ============================================================================================================
# PROJECT      : STEA — Science, Technology & Energy Analysis
# PIPELINE     : MSTI — Main Science and Technology Indicators
# MODULE       : pipelines/utils/msti_system_utils.py
# PURPOSE      : Unified system diagnostics, GPU utilities, and performance benchmarking
# DESCRIPTION  :
#   - Standardized warning formatting for consistent error display
#   - CPU, RAM, and GPU resource monitoring utilities
#   - GPU memory management and cleanup helpers (CuPy)
#   - GPU benchmarking for KNN complexity analysis
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import logging   # standardized logging
import platform   # system information (CPU architecture, OS)
import shutil   # check presence of NVIDIA utilities
import subprocess   # execute system commands (e.g., nvidia-smi)
import time   # benchmarking and performance measurement
import warnings   # manage and override Python warnings
from typing import Callable, List, Optional   # type hints

# 1.2 --- Third-party libraries ---
import cupy as cp   # GPU-accelerated arrays (RAPIDS)
import numpy as np   # CPU-based numerical operations
import pandas as pd   # tabular data summary and outputs
import psutil   # CPU / RAM usage statistics


# ------------------------------------------------------------------------------------------------------------
# 2. Warning Management
# ------------------------------------------------------------------------------------------------------------
def show_warning(message, category, filename, lineno, file=None, line=None):
    """
    Standardized Python warning display with simplified formatting.

    Provides consistent warning output across the MSTI pipeline by formatting 
    warnings in a concise single-line format with file location and line number.

    Parameters
    ----------
    message : str
        Warning message content.
    category : Warning
        Warning type (e.g., UserWarning, RuntimeWarning).
    filename : str
        File name where the warning was triggered.
    lineno : int
        Line number where the warning occurred.
    file : object, optional
        Output stream (not used in this implementation).
    line : str, optional
        Source line content (not used in this implementation).

    Returns
    -------
    None
    """
    print(f"\nWarning: {message} (line {lineno} in {filename})")


def init_warning_hook():
    """
    Activate the global uniform warning hook for the MSTI pipeline.

    Should be called explicitly in the main pipeline orchestrator before 
    execution to standardize all Python warnings throughout the processing 
    workflow.

    Returns
    -------
    None
    """
    warnings.showwarning = show_warning


# ------------------------------------------------------------------------------------------------------------
# 3. Formatting Helpers
# ------------------------------------------------------------------------------------------------------------
def format_bytes(n_bytes: int) -> str:
    """
    Convert a file size in bytes into a human-readable string (B, KB, MB, GB, TB).

    Parameters
    ----------
    n_bytes : int
        Raw file size in bytes.

    Returns
    -------
    str
        Formatted file size using SI units with 2 decimal precision.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:   # iterate units
        if n_bytes < 1024:   # threshold for next unit
            return f"{n_bytes:.2f} {unit}"   # formatted size
        n_bytes /= 1024   # scale down for next unit
    return f"{n_bytes:.2f} PB"   # petabyte fallback


# ------------------------------------------------------------------------------------------------------------
# 4. GPU Memory Management
# ------------------------------------------------------------------------------------------------------------
def release_gpu_memory():
    """
    Force-release GPU memory cache from CuPy default memory pools.

    Useful to free GPU memory between large computations or iterative 
    processes to prevent out-of-memory errors. Clears both device memory 
    and pinned host memory.

    Returns
    -------
    None
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ------------------------------------------------------------------------------------------------------------
# 5. System Information and Resource Monitoring
# ------------------------------------------------------------------------------------------------------------
def display_system_info(wait: bool = False, return_str: bool = False) -> Optional[str]:
    """
    Display comprehensive system hardware information for diagnostics.

    Reports processor specifications, memory availability, and GPU status including
    VRAM utilization and compute capability (via nvidia-smi for NVIDIA GPUs).

    Parameters
    ----------
    wait : bool, default=False
        If True, waits one second to compute average CPU usage instead of instantaneous.
    return_str : bool, default=False
        If True, returns formatted string instead of printing to console.

    Returns
    -------
    str or None
        Formatted diagnostic string if return_str=True, otherwise None.
    """
    lines = []   # accumulator for output lines
    
    # 5.1 --- CPU information ---
    lines.append("\n=== CPU ===")
    cpu_name = platform.processor() or platform.machine()   # get processor name with fallback
    lines.append(f"Processor Name          : {cpu_name}")
    lines.append(f"Physical Cores          : {psutil.cpu_count(logical=False)}")
    lines.append(f"Logical Cores           : {psutil.cpu_count(logical=True)}")
    
    freq = psutil.cpu_freq()   # get CPU frequency
    if freq:
        lines.append(f"CPU Frequency           : {freq.current:.2f} MHz")
    else:
        lines.append("CPU Frequency           : N/A")
    
    cpu_interval = 1 if wait else 0   # wait duration for CPU stats
    lines.append(f"CPU Usage               : {psutil.cpu_percent(interval=cpu_interval):.1f} %")

    # 5.2 --- RAM information ---
    ram = psutil.virtual_memory()   # get memory statistics
    lines.append("\n=== RAM ===")
    lines.append(f"Total RAM               : {ram.total / 1e9:.2f} GB")
    lines.append(f"Available RAM           : {ram.available / 1e9:.2f} GB")
    lines.append(f"RAM Usage               : {ram.percent:.1f} %")

    # 5.3 --- GPU information ---
    lines.append("\n=== GPU (NVIDIA) ===")
    try:
        if shutil.which("nvidia-smi"):   # verify nvidia-smi utility availability
            gpu_info = subprocess.run(
                ["nvidia-smi"],   # execute GPU info command
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2   # reduced from 5s to 2s
            ).stdout   # capture command output
            lines.append(gpu_info)
        else:
            lines.append("No NVIDIA GPU detected or 'nvidia-smi' not found.")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        lines.append("Unable to retrieve GPU information (timeout or missing drivers).")
    
    output = "\n".join(lines)   # join all lines
    
    if return_str:
        return output   # return string
    else:
        print(output)   # print to console
        return None


def monitor_resources(bar_length: int = 30, return_str: bool = False) -> Optional[str]:
    """
    Display ASCII-based visualization of system resource usage.

    Monitors CPU memory (RAM) and GPU memory utilization with visual progress bars,
    useful for tracking resource consumption before and after heavy GPU computations
    such as KNN imputation or matrix operations.

    Parameters
    ----------
    bar_length : int, default=30
        Length of the ASCII progress bar for usage ratio visualization.
    return_str : bool, default=False
        If True, returns formatted string instead of printing to console.

    Returns
    -------
    str or None
        Formatted resource string if return_str=True, otherwise None.
    """
    # 6.1 --- CPU RAM monitoring ---
    ram = psutil.virtual_memory()   # get current memory statistics
    ram_used = ram.used / 1e9   # convert to gigabytes
    ram_total = max(ram.total / 1e9, 1e-6)   # total RAM with safety margin
    ram_ratio = ram_used / ram_total   # compute usage ratio

    # 6.2 --- GPU CUDA monitoring ---
    try:
        cp.get_default_memory_pool().free_all_blocks()   # clear CuPy memory cache for accurate reading
        gpu_free, gpu_total = cp.cuda.Device().mem_info   # get GPU memory info
        gpu_used = (gpu_total - gpu_free) / 1e9   # convert to gigabytes
        gpu_total_gb = gpu_total / 1e9   # total GPU memory in GB
        gpu_ratio = gpu_used / max(gpu_total_gb, 1e-6)   # compute usage ratio
        
        smi_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2   # reduced from 5s to 2s
        ).stdout.strip()   # query GPU utilization percentage
        gpu_util = float(smi_out.split("\n")[0]) if smi_out else 0.0   # parse utilization value
    except (cp.cuda.runtime.CUDARuntimeError, FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        gpu_used, gpu_total_gb, gpu_ratio, gpu_util = 0.0, 1.0, 0.0, 0.0   # fallback values if GPU unavailable

    # 6.3 --- Generate ASCII progress bars ---
    def _progress_bar(ratio: float) -> str:
        """
        Generate ASCII progress bar for resource usage visualization.
        
        Parameters
        ----------
        ratio : float
            Usage ratio between 0.0 and 1.0.
        
        Returns
        -------
        str
            Formatted ASCII progress bar (filled █ + empty -).
        """
        filled = int(ratio * bar_length)
        return "█" * filled + "-" * (bar_length - filled)

    # 6.4 --- Format diagnostics ---
    lines = []   # accumulator for output lines
    lines.append("\n=== Memory Diagnostics ===")
    lines.append(f"RAM Usage : |{_progress_bar(ram_ratio)}| {ram_used:.2f}/{ram_total:.2f} GB ({ram_ratio * 100:.1f}%)")
    gpu_usage_str = (
        f"GPU Usage : |{_progress_bar(gpu_ratio)}| "
        f"{gpu_used:.2f}/{gpu_total_gb:.2f} GB ({gpu_ratio * 100:.1f}%) "
        f"[Utilization: {gpu_util:.0f}%]"
    )
    lines.append(gpu_usage_str)
    lines.append("-" * min(bar_length + 40, 100))   # visual separator line
    
    output = "\n".join(lines)   # join all lines
    
    if return_str:
        return output   # return string
    else:
        print(output)   # print to console
        return None


# ------------------------------------------------------------------------------------------------------------
# 7. System Snapshot Logger
# ------------------------------------------------------------------------------------------------------------
def log_system_snapshot(
    label: str,
    logger: Optional[logging.Logger] = None,   # ✅ type hint explicite
    wait_cpu: bool = False,
    bar_length: int = 30
) -> str:
    """
    Capture and log comprehensive system diagnostics snapshot.
    
    Combines CPU, RAM, GPU status, and memory usage into unified diagnostic report.
    Useful for monitoring resource utilization before/after intensive computations.
    
    Parameters
    ----------
    label : str
        Descriptive label (e.g., "Before Imputation", "After Training").
    logger : logging.Logger, optional
        Logger instance for output. If None, returns string without logging.
    wait_cpu : bool, default=False
        If True, waits 1 second for accurate CPU usage measurement.
    bar_length : int, default=30
        Length of ASCII progress bars for memory visualization.
    
    Returns
    -------
    str
        Formatted diagnostic snapshot string.
    """
    lines = []
    lines.append(f"=== {label} ===")
    lines.append("")
    
    sys_info = display_system_info(wait=wait_cpu, return_str=True)
    if sys_info:
        lines.append(sys_info)
    
    mem_info = monitor_resources(bar_length=bar_length, return_str=True)
    if mem_info:
        lines.append(mem_info)
    
    output = "\n".join(lines)
    
    if logger:
        logger.info(output)
    
    return output


# ------------------------------------------------------------------------------------------------------------
# 8. GPU Benchmark — KNN Imputation Complexity (Optimized)
# ------------------------------------------------------------------------------------------------------------
def benchmark_knn_gpu_complexity(
    knn_func: Callable[[cp.ndarray, int, int], cp.ndarray],
    data: np.ndarray,
    k_values: List[int],
    n_iterations: int = 3,
    n_list: Optional[List[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Benchmark empirical runtime complexity of the GPU KNN imputer.

    Tests the imputer across varying sample sizes (n_list) and neighbor counts (k_values)
    to estimate computational complexity and identify optimal parameters.

    Parameters
    ----------
    knn_func : callable
        GPU-based imputation function with signature (data_gpu, k, n_iterations) -> imputed_gpu.
    data : numpy.ndarray
        Full numeric dataset to subsample from for benchmarking.
    k_values : list of int
        Neighbor counts to benchmark.
    n_iterations : int, default=3
        Number of refinement iterations for each test.
    n_list : list of int, optional
        List of subsample sizes for complexity analysis.
        If None, defaults to [len(data)] (constant n).
    verbose : bool, default=True
        Whether to print progress messages to console.

    Returns
    -------
    pandas.DataFrame
        Benchmark results with columns: ['n_samples', 'k_neighbors', 'time_s', 'relative_speedup']
    """
    # 8.1 --- Initialize parameters ---
    if n_list is None:
        n_list = [len(data)]   # fallback: single full dataset

    results = []   # store benchmark results
    t0_global = time.time()   # global timer

    try:
        # 8.2 --- Iterate over sample sizes ---
        for n in n_list:
            sample = data[:n, :]
            sample_gpu = cp.asarray(sample, dtype=cp.float64)
            
            if verbose:
                print(f"[GPU Benchmark] n={n:,} samples")
            
            # 8.2.1 --- Iterate over k values ---
            for k in k_values:
                t0 = time.time()
                try:
                    _ = knn_func(sample_gpu, k=k, n_iterations=n_iterations)
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.time() - t0
                    results.append({"n_samples": n, "k_neighbors": k, "time_s": elapsed})
                    if verbose:
                        print(f"  • k={k:<2} | time={elapsed:.4f}s")
                except cp.cuda.memory.OutOfMemoryError:
                    print(f"[WARN] OOM at n={n}, k={k}")   # log OOM error
                    results.append({"n_samples": n, "k_neighbors": k, "time_s": np.nan})   # store NaN
                    continue   # skip to next iteration
    
    finally:
        release_gpu_memory()   # guaranteed cleanup

    # 8.3 --- Compute relative speedup ---
    df_bench = pd.DataFrame(results)   # convert to DataFrame
    min_time = df_bench["time_s"].min()   # find minimum time
    df_bench["relative_speedup"] = (min_time / df_bench["time_s"]).round(2)   # compute speedup
    df_bench["relative_speedup"] = df_bench["relative_speedup"].fillna(0)  # fill NaN speedup with 0

    # 8.4 --- Log summary ---
    if verbose:
        total = time.time() - t0_global   # total benchmark time
        print(f"[GPU Benchmark Completed] {len(results)} runs in {total:.2f}s\n")

    return df_bench   # return benchmark results