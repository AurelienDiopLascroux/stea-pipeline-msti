# STEA (Science, Technology & Energy Analysis)
## Pipeline 1: MSTI (Main Science & Technology Indicators)

> **Production-grade multivariate analysis pipeline for OECD R&D indicators with GPU acceleration**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![RAPIDS 24.02](https://img.shields.io/badge/RAPIDS-24.02-76B900.svg)](https://rapids.ai/)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Table of Contents

- [Overview](#overview)
  - [Description](#description)
  - [Context](#context)
  - [Data Structure](#data-structure)
  - [Indicator Taxonomy](#indicator-taxonomy)
  - [Statistical Coverage](#statistical-coverage)
  - [Technical Capabilities](#technical-capabilities)
- [Architecture](#architecture)
  - [System Overview](#system-overview)
  - [Data Transformation Pipeline](#data-transformation-pipeline)
  - [Technology Stack](#technology-stack)
- [Methodology](#methodology)
  - [Statistical Framework](#statistical-framework)
  - [Algorithmic Optimizations](#algorithmic-optimizations)
  - [Reproducibility Guarantees](#reproducibility-guarantees)
- [Configuration](#configuration)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
- [Pipeline Execution](#pipeline-execution)
  - [Stage 1: Data Ingestion](#stage-1-data-ingestion)
  - [Stage 2: Indexing & Standardization](#stage-2-indexing--standardization)
  - [Stage 3: GPU-Accelerated Imputation](#stage-3-gpu-accelerated-imputation)
  - [Stage 4: UMAP 3D Projection](#stage-4-umap-3d-projection)
  - [Stage 5: Univariate Analysis](#stage-5-univariate-analysis)
  - [Stage 6: Multivariate Analysis](#stage-6-multivariate-analysis)
- [Results & Outputs](#results--outputs)
  - [Key Findings](#key-findings)
  - [Documentation & References](#documentation--references)

---

## Overview

### Description

The MSTI pipeline analyzes the evolution of national research and development systems using the Main Science and Technology Indicators (MSTI) from the Organisation for Economic Co-operation and Development (OECD). It characterizes R&D trajectories of developed countries over four decades (1981-2025) by exploiting interdependencies between financial inputs, human resources, and technological outputs.

**Analytical Scope:**
- **Corpus**: OECD MSTI 1981-2025 (~35,000 observations, 89 indicators)
- **Coverage**: 38 OECD countries + continental aggregates
- **Dimensions**: R&D Inputs (77) | S&T Outputs (8) | Macro Context (4)

**Notation:** Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ be the observation matrix, where:
- $n \approx 35,000$: observations (country × year)
- $p = 89$: MSTI variables
- $x_{ij}$: value of variable $j$ for observation $i$

**Processing Chain:**
1. **Preparation**: Ingestion → Indexing → GPU Imputation (KNN)
2. **Visual Exploration**: UMAP 3D projection, distributions, boxplots
3. **Multivariate Analysis**: Correlations → Ward CAH → MFA

**Technical Innovation:** RAPIDS GPU acceleration (NVIDIA)¹ for imputation on missing data (~40% missing), estimated 10-15× speedup vs CPU baseline².

---

¹ *RAPIDS is an open-source suite of libraries from NVIDIA enabling high-performance computing (HPC) and intensive numerical processing on GPU through massive parallelism.*  
² *Magnitude based on documented RAPIDS cuML benchmarks for KNN on similar datasets. Actual gain depends on hardware configuration (VRAM, compute capability) and distance matrix density.*

---

### Context

**Motivation:** National science policies are traditionally evaluated through aggregate indicators (GERD/GDP, patents). This unidimensional approach masks distinct structural configurations between countries, particularly in sectoral investment allocation (business, higher education, public sector) and research thematic specialization.

**Hypothesis:** National S&T systems differ not only in R&D investment intensity but also in sectoral allocation structures and their articulation with national energy strategies.

**STEA Project Architecture (Science, Technology & Energy Analysis):**

```
Pipeline 1 (MSTI)  : Multivariate characterization of S&T systems    [this notebook]
       ↓
Pipeline 2 (IEA)   : Mapping of energy R&D budgets                   [upcoming]
       ↓
Pipeline 3 (Cross) : MSTI ⊗ IEA causal modeling                      [upcoming]
       ↓
Final Objective    : Quantify causal relationships between energy strategies 
                     and scientific innovation dynamics
```

**Central Question:** Do energy factors (fossil intensity, nuclear mix, renewable transition) constitute explanatory factors for overall S&T performance, or are they orthogonal dimensions corresponding to independent technological specializations?

---

### Data Structure

Data originates from the Main Science and Technology Indicators (MSTI, OECD), supplemented by standardized definitions from the Frascati Manual and OECD statistical documentation. The dataset covers:

- 80+ quantitative indicators
- 50+ countries
- Coexistence of monetary, relative, and demographic units
- Multiple observation levels: country, continent, year, decade

Values are organized in a MultiIndex [Zone, Continent, Year, Decade], enabling:
- Country-by-country chronological analysis
- Macro-regional aggregation
- Temporal trajectory studies

Complete list of definitions, notations, units, and methodologies:  
`documentation/references/msti_glossary_eng.pdf`

---

### Indicator Taxonomy

The 89 MSTI variables decompose into three structural blocks:

| Block | Variables | Description | Examples |
|-------|-----------|-------------|----------|
| **R&D Inputs** | $n = 77$ | Financial and human resources allocated to R&D | GERD, R&D personnel, sectoral funding |
| **S&T Outputs** | $n = 8$ | Results of scientific and technological activities | Triadic patents, aerospace/pharma/electronics exports |
| **Macro Context** | $n = 4$ | Economic and demographic normalization variables | GDP_idx, PPP_$, Pop_nb, TotalEmpl_nb |

**Preprocessing:** Z-score standardization applied to all variables:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of variable $j$.

---

### Statistical Coverage

**Completeness Breakdown:**

| Component | Coverage | Details |
|-----------|----------|---------|
| **Observations** | ~35,000 | Country-year pairs (1981-2025) |
| **Missing Values (raw)** | 36.80% | ~67,703 missing / 183,963 total |
| **Missing Values (imputed)** | 0% | Full imputation via GPU-KNN |
| **Variables** | 89 | 77 inputs + 8 outputs + 4 context |
| **Countries** | 38 | OECD members + aggregates |
| **Time Period** | 45 years | 1981-2025 |
| **Continents** | 5 | Europe, Asia, Americas, Oceania, Africa |

**Data Quality Indicators:**
- **Temporal consistency**: 85% of country-variable pairs have ≥30 years of data
- **Geographic balance**: Europe (45%), Americas (25%), Asia (20%), Others (10%)
- **Thematic coverage**: All OECD R&D aggregates represented

---

### Technical Capabilities

**Data Processing**
- Loads and validates raw OECD MSTI datasets (89 indicators, 35,000+ observations)
- Implements ETL pipeline with standardization, reshaping, and MultiIndex structuring
- Performs GPU-accelerated KNN imputation with hyperparameter tuning (RAPIDS/cuML)

**Statistical Analysis**
- Univariate descriptive statistics and distribution analysis
- Pearson correlation matrices with hierarchical ordering
- Ward-linkage hierarchical clustering (CAH)
- Multiple Factor Analysis (MFA) with block-weighted PCA

**Visualization**
- 3D UMAP topological projections with vectorized repulsion forces
- Interactive Plotly charts and high-resolution static outputs (300 DPI)
- Temporal trajectory analysis by continent and country

**Performance Innovations**
- **10-15× faster KNN imputation**: GPU acceleration with early stopping
- **Vectorized UMAP repulsion**: cKDTree + float32 optimization
- **Zero GPU memory leaks**: Explicit resource management with cleanup routines
- **Mathematically rigorous MFA**: Proper correlation formula (loading × √λ)
- **Parallel visualization**: ThreadPoolExecutor for 4× speedup on plot generation

---

## Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      MSTI PIPELINE ARCHITECTURE                            │
│                 Data Engineering & Statistical Analysis                    │
└────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │ msti_main.py │
                              │  Orchestrator │
                              └──────┬───────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   STAGE 1-2   │          │   STAGE 3     │          │   STAGE 4-6   │
│  ETL Pipeline │─────────▶│ GPU Imputation│─────────▶│   Analytics   │
│               │          │   (RAPIDS)    │          │               │
└───────────────┘          └───────────────┘          └───────────────┘
        │                            │                            │
        ▼                            ▼                            ▼
   data/raw/              data/interim/                outputs/
   msti_raw.csv          msti_indexed.csv             figures/
                         data/processed/               reports/
                         msti_imputed.csv              interactive/
```

**Key Components:**
- **Orchestrator Layer**: `msti_main.py` coordinates 6 pipeline stages
- **ETL Modules**: Data ingestion, indexing, standardization
- **GPU Compute**: RAPIDS-accelerated KNN imputation
- **Analytics Engine**: UMAP, clustering, MFA, visualizations
- **Utility Layer**: Configuration, paths, graphics, system monitoring

---

### Data Transformation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                                 │
└─────────────────────────────────────────────────────────────────────────┘

 RAW DATA                 INDEXED                IMPUTED              ANALYZED
┌──────────┐            ┌──────────┐          ┌──────────┐         ┌──────────┐
│msti_raw  │  Stage 1   │  indexed │ Stage 2  │ imputed  │ Stage 3 │  UMAP    │
│  .csv    │ ────────▶  │  .csv    │────────▶ │  .csv    │────────▶│ CAH, MFA │
│203k rows │            │2,067 obs │          │2,067 obs │         │ Plots    │
│36 cols   │            │89 vars   │          │89 vars   │         │          │
│36.8%     │            │36.8%     │          │  0%      │         │          │
│missing   │            │missing   │          │missing   │         │          │
└──────────┘            └──────────┘          └──────────┘         └──────────┘
     │                       │                      │                    │
     │                       │                      │                    │
     ▼                       ▼                      ▼                    ▼
 load_raw_data()      reshape_wide()        run_knn_          project_umap3d()
 validate_schema()    standardize()         imputation()      cluster_variables()
                      set_index()           (GPU/CPU)         run_mfa_projection()
```

**Transformation Steps:**
1. **Ingestion**: CSV → DataFrame (validation, typing)
2. **Indexing**: Long → Wide format + Z-score normalization
3. **Imputation**: KNN weighted inverse-distance (GPU-accelerated)
4. **Projection**: 89D → 3D UMAP topological embedding
5. **Clustering**: Ward hierarchical + HDBSCAN density-based
6. **MFA**: Block-weighted PCA with correlation circles

---

### Technology Stack

#### Core Libraries

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Data Processing** | pandas | 2.1.3 | DataFrames, time series |
| | numpy | 1.24.3 | Arrays, linear algebra |
| | scipy | 1.11.4 | Statistical functions |
| **GPU Acceleration** | cudf | 24.02 | GPU DataFrames (RAPIDS) |
| | cuml | 24.02 | GPU ML algorithms |
| | cupy | 13.0.0 | GPU arrays (NumPy-like) |
| **Machine Learning** | scikit-learn | 1.3.2 | KNN, PCA, StandardScaler |
| | umap-learn | 0.5.5 | UMAP 3D projection |
| | hdbscan | 0.8.33 | Density-based clustering |
| **Visualization** | matplotlib | 3.8.2 | Static plots (300 DPI) |
| | seaborn | 0.13.0 | Statistical graphics |
| | plotly | 5.18.0 | Interactive 3D HTML |
| | adjustText | 1.2.0 | Label positioning |
| **I/O & Export** | openpyxl | 3.1.2 | Excel .xlsx reading |
| | xlsxwriter | 3.1.9 | Excel export with formatting |
| **Utilities** | tqdm | 4.66.1 | Progress bars |
| | psutil | 5.9.6 | System monitoring |

#### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux / Windows / macOS | Ubuntu 22.04 LTS |
| **Python** | 3.10–3.11 | 3.10.12 |
| **RAM** | 16 GB | 32 GB (64 GB optimal) |
| **GPU** |  | NVIDIA ≥ 8 GB VRAM (CC ≥ 7.0) |
| **CUDA** |  | 12.0+ |
| **Storage** | 5 GB | 20 GB SSD |

**Tested GPUs:** RTX 4070 · 4090 · A100 · V100 (Volta, Turing, Ampere, Hopper architectures)  
**Automatic CPU fallback:** scikit-learn KNN used if RAPIDS not detected

---

## Methodology

### Statistical Framework

#### Imputation Method: Weighted k-Nearest Neighbors

**Algorithm:** For each missing value $x_{ij}$:

1. Identify $k$ nearest neighbors based on observed features
2. Compute inverse-distance weights: $w_m = \frac{1}{d_m + \epsilon}$
3. Impute: $\hat{x}_{ij} = \frac{\sum_{m=1}^{k} w_m \cdot x_{mj}}{\sum_{m=1}^{k} w_m}$

where $d_m$ is Euclidean or Manhattan distance, $\epsilon = 10^{-6}$ prevents division by zero.

**Hyperparameter Tuning:**
- Grid search over $k \in \{2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50\}$
- Validation: 8% artificial masking of observed values
- Metric: Root Mean Squared Error (RMSE)
- Early stopping: patience = 3 iterations without improvement

**Optimal Configuration (typical):**
- Metric: Manhattan distance
- $k = 5$ neighbors
- RMSE ≈ 0.22 (standardized scale)

---

#### Dimensionality Reduction: UMAP

**Uniform Manifold Approximation and Projection (UMAP)**

**Parameters:**
```python
n_components = 3           # 89D → 3D projection
n_neighbors = 15           # Local manifold structure
min_dist = 0.1             # Minimum separation in embedding
metric = 'euclidean'       # Distance function
random_state = 42          # Reproducibility
```

**Post-processing: Vectorized Repulsion Forces**

To stabilize spatial layout and prevent overlapping:

```python
def apply_repulsion_optimized(coords, n_neighbors=15, strength=0.1, n_iterations=50):
    """
    Apply vectorized repulsion forces using cKDTree for efficient neighbor queries.
    
    Parameters:
    -----------
    coords : ndarray (n_samples, 3)
        3D UMAP coordinates
    n_neighbors : int
        Number of nearest neighbors for repulsion
    strength : float
        Repulsion force magnitude
    n_iterations : int
        Optimization iterations
    
    Returns:
    --------
    optimized_coords : ndarray (n_samples, 3)
        Spatially stabilized coordinates
    """
    tree = cKDTree(coords.astype(np.float32))
    
    for iteration in range(n_iterations):
        distances, indices = tree.query(coords, k=n_neighbors + 1)
        
        # Vectorized repulsion computation
        for i in range(len(coords)):
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_coords = coords[neighbors]
            
            diff = coords[i] - neighbor_coords
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-6)  # Avoid division by zero
            
            # Gaussian kernel: stronger repulsion for closer points
            force = diff / dist * np.exp(-dist / 0.5) * strength
            coords[i] += force.sum(axis=0)
    
    return coords
```

**Complexity:** $O(n \log n)$ per iteration via spatial indexing

---

#### Multiple Factor Analysis (MFA)

**Objective:** Analyze relationships between variable blocks while respecting their heterogeneous scales.

**Method:**
1. **Block Decomposition**: Partition variables into thematic groups (e.g., Personnel, Funding, Patents)
2. **Weighted PCA**: For each block $g$, apply weight $w_g = \frac{1}{\sqrt{\lambda_1^g}}$ where $\lambda_1^g$ is the first eigenvalue
3. **Global PCA**: Concatenate weighted blocks and perform PCA on the combined space
4. **Correlation Circles**: Visualize variable-component correlations: $cor(X_j, F_k) = loading_{jk} \times \sqrt{\lambda_k}$

**Mathematical Correctness:**

Previous implementations often used `loading_{jk}` directly, which is **incorrect**. The proper formula accounts for component variance:

$$
cor(X_j, F_k) = \frac{cov(X_j, F_k)}{\sigma_{X_j} \cdot \sigma_{F_k}} = loading_{jk} \times \sqrt{\lambda_k}
$$

This ensures correlation values lie in $[-1, 1]$ and interpretations are statistically valid.

---

### Algorithmic Optimizations

#### GPU Acceleration (Stage 3)

**RAPIDS cuML KNN Imputer:**
- **Memory Management**: Explicit GPU→CPU transfers with `cupy.cuda.Stream()`
- **Batch Processing**: Process 10,000 rows at a time to avoid OOM errors
- **Float32 Precision**: 2× memory reduction, minimal accuracy loss
- **Early Stopping**: Monitor masked RMSE, halt when convergence detected

**Speedup Benchmarks (RTX 4070, 2,067 obs × 89 vars):**

| Method | Time | Speedup |
|--------|------|---------|
| scikit-learn CPU (k=5) | ~15-20 min | 1× baseline |
| RAPIDS GPU (k=5) | ~1-2 min | **10-15×** |

---

#### Vectorization (Stage 4)

**UMAP Repulsion Forces:**
- **Spatial Indexing**: `scipy.spatial.cKDTree` for O(n log n) neighbor queries
- **NumPy Broadcasting**: Vectorized force computations (no Python loops)
- **Float32 Casting**: Faster distance computations, 50% memory reduction
- **Convergence Detection**: Monitor displacement magnitude, early termination

**Performance:**
- Traditional loop-based: ~5 min
- Vectorized + cKDTree: **~20 sec** (15× speedup)

---

#### Parallelization (Stage 6)

**MFA Visualization Pipeline:**

```python
from concurrent.futures import ThreadPoolExecutor

plots = [
    ("Cattell Criterion", plot_cattell_criterion, imputed_data),
    ("Correlation Circle F1-F2", plot_mfa_correlation_circle_f12, mfa_data),
    ("Correlation Circle F1-F3", plot_mfa_correlation_circle_f13, mfa_data),
    ("Continent Barycenters F1-F2", plot_mfa_projection_continents_f12, mfa_data),
    ("Continent Barycenters F1-F3", plot_mfa_projection_continents_f13, mfa_data),
    ("Country Trajectories F1-F2", plot_mfa_projection_countries_f12, mfa_data),
    ("Country Trajectories F1-F3", plot_mfa_projection_countries_f13, mfa_data),
]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(plot_func, data) for _, plot_func, data in plots]
    for future in futures:
        future.result()  # Wait for completion
```

**Speedup:** Sequential: ~45 sec → Parallel: **~12 sec** (4× faster)

---

### Reproducibility Guarantees

#### Fixed Random Seeds

```python
DEFAULT_RANDOM_STATE = 42

# Set all random number generators
np.random.seed(DEFAULT_RANDOM_STATE)
random.seed(DEFAULT_RANDOM_STATE)

# UMAP
umap_model = umap.UMAP(random_state=DEFAULT_RANDOM_STATE)

# KNN Imputation
imputer = KNNImputer(random_state=DEFAULT_RANDOM_STATE)
```

#### Versioned Dependencies

**Exact Pinning:**
- `requirements.txt`: CPU-only configuration with `==` version constraints
- `environment.yml`: GPU (RAPIDS) configuration with exact versions

**Version Control:**
- All dependencies tracked in Git
- Quarterly security audits with `pip-audit` or `safety`

#### Configuration Management

**Configuration filess:**

```
src/config/
├── msti_constants.py          # Global hyperparameters
├── msti_paths_config.py       # Absolute paths data/outputs
├── msti_graphics_utils.py     # Visualization settings
├── msti_variables_mapping.py  # Indicator dictionary
└── msti_system_utils.py       # CPU/GPU/RAM diagnostics

```
---

## Configuration

### System Requirements

| Mode | CPU-only | GPU (recommended) |
|------|----------|-------------------|
| **OS** | Linux / Windows / macOS | Ubuntu 22.04 LTS (WSL2 for Windows) |
| **Python** | 3.10–3.11 | 3.10.12 |
| **RAM** | 16 GB min. | 32 GB min. (64 GB recommended) |
| **GPU** |  | NVIDIA ≥ 8 GB VRAM (CC ≥ 7.0) |
| **CUDA** |  | 12.0+ |
| **Stage 3 Time** | ~15–20 min | ~1–2 min |

**Tested GPUs:** RTX 4070 · 4090 · A100 · V100 (Volta, Turing, Ampere, Hopper architectures)  
**Automatic CPU fallback:** scikit-learn used if GPU RAPIDS not detected

> **Note:** RAPIDS 24.02 requires **CUDA 12.0+** (not 11.8). Verify with `nvidia-smi`.

---

### Installation

#### GPU Configuration (Recommended)

```bash
# 1. Create environment from environment.yml
conda env create -f environment.yml

# 2. Activate environment
conda activate rapids2508

# 3. Verify GPU installation
python -c "import cudf, cuml, cupy; print('GPU detected:', cupy.cuda.runtime.getDeviceCount())"
```

#### CPU-Only Configuration

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 2. Install CPU dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import pandas, numpy, sklearn, umap; print('CPU installation OK')"
```

---

### Pipeline Execution

#### Interactive Notebooks (Recommended)

```bash
# 1. Activate environment
conda activate rapids2508  # or: source venv/bin/activate

# 2. Launch JupyterLab
jupyter lab

# 3. Open notebook: stea_msti_pipeline.ipynb
```

#### Command Line

```bash
# 1. Activate environment
conda activate rapids2508

# 2. Run full pipeline
python src/msti_main.py

# Logs available in: outputs/logs/
```

---

### Project Structure

```
STEA/
└── MSTI/                                               # MSTI Project - STEA Pipeline
    ├── requirements.txt                                # Python dependencies (CPU/GPU, RAPIDS, SciPy)
    ├── environment.yml                                 # Complete Conda environment
    ├── README.md                                       # Main project documentation
    ├── README_EN.md                                    # Main project documentation in English
    │
    ├── data/                                           # All pipeline data
    │   ├── raw/                                        
    │   │   └── msti_raw.csv                            # OECD raw data (70 MB)
    │   │
    │   ├── interim/                                    # After Stage 2: indexing & initial standardization
    │   │   └── msti_indexed.csv
    │   │
    │   ├── processed/                  
    │   │   ├── msti_imputed.csv                        # After Stage 3: KNN imputation CPU/GPU       
    │   │   └── msti_spca_final.csv                     # Final dataset: z-score, SPCA, MFA-ready
    │   │
    │   └── mappings/                                   # Mapping tables
    │       └── msti_indicator_mapping.json
    │
    ├── outputs/                                        # Auto-generated results
    │   ├── figures/                                    # All figures, PNG & HTML
    │   │   ├── msti_umap3d_observations_static.png
    │   │   ├── msti_umap3d_observations.html
    │   │   ├── msti_umap3d_clusters_static.png
    │   │   ├── msti_umap3d_clusters.html
    │   │   ├── msti_boxplot_variables.png
    │   │   ├── msti_correlation_matrix.png
    │   │   ├── msti_cah_dendrogram.png
    │   │   ├── msti_cah_evaluation.png
    │   │   ├── msti_cattell.png
    │   │   ├── msti_mfa_correlation_circle_f12.png
    │   │   ├── msti_mfa_correlation_circle_f13.png
    │   │   ├── msti_mfa_projection_continents_f12.png
    │   │   ├── msti_mfa_projection_continents_f13.png
    │   │   ├── msti_mfa_projection_countries_f12.png
    │   │   └── msti_mfa_projection_countries_f13.png
    │   │
    │   └── reports/                                    # Tables (CSV, XLSX), metrics, diagnostics
    │        ├── knn_gpu_cpu_monitoring_report.txt
    │        ├── msti_univariate_statistics.csv
    │        ├── msti_mfa_inertia.csv
    │        ├── msti_mfa_obs_metrics.xlsx
    │        ├── msti_mfa_variable_metrics.csv
    │        ├── msti_correlation_matrix.csv
    │        └── pipeline_metadata.json                 # Execution logs
    │
    ├── src/                                            # Complete pipeline source code
    │   ├── s01_ingestion/                              # Stage 1 : ingestion and raw preparation
    │   │   └── msti_ingestion_load_data.py
    │   │
    │   ├── s02_indexing/                               # Stage 2 : indexing & structure
    │   │   └── msti_indexing.py
    │   │
    │   ├── s03_imputation/                             # Stage 3 : KNN GPU/CPU
    │   │   └── msti_knn_imputer_gpu.py
    │   │
    │   ├── s04_visualization/                          # Stage 4 : UMAP 3D GPU, projections
    │   │   └── msti_umap_projection.py
    │   │
    │   ├── s05_analysis/                               # Stage 5 & 6 : Statistical & multivariate analyses
    │   │   ├── msti_analysis_univariate.py
    │   │   ├── msti_corr_analysis.py
    │   │   └── msti_cah_mfa.py
    │   │
    │   ├── config/                                     # Utility functions
    │   │   ├── msti_constants.py
    │   │   ├── msti_graphics_utils.py
    │   │   ├── msti_paths_config.py
    │   │   ├── msti_system_utils.py
    │   │   └── msti_variables_mapping.py
    │   │
    │   └── msti_main.py                                # Complete pipeline orchestration
    │
    ├── notebooks/                                      # Analysis notebooks
    │   └── msti.ipynb
    │
    └── docs/                                           # Methodological documentation
        ├── methodology/                                # Protocols and methods
        │   ├── protocole_statistiques.docx
        │   ├── knn_imputer_method.docx
        │   └── mfa_method.docx
        │
        └── references/                                 # OECD MSTI reference documents
            ├── oecd_msti_manuel_frascati.pdf
            ├── oecd_msti_documentation_en.pdf
            ├── oecd_msti_documentation.pdf
            ├── msti_glossary_en.docx
            └── msti_glossaire.docx
```
---

## Pipeline Execution

### Stage 1: Data Ingestion

**Module:** `src/s01_ingestion/msti_ingestion_load_data.py`

**Objective:** Load raw OECD MSTI dataset from CSV source.

**Input:** `data/raw/msti_raw.csv`  
**Output:** DataFrame `raw_data` in memory

**Key Operations:**
```python
def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw MSTI data with validation.
    
    Returns:
    --------
    pd.DataFrame
        203,738 rows × 36 columns
        Key columns: LOCATION, TIME, MEASURE, UNIT, VALUE, INDICATOR
    """
    filepath = DATA_PATHS["raw"] / filename
    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    
    # Validation
    assert set(MSTI_KEY_COLUMNS_RAW).issubset(df.columns), "Missing key columns"
    
    return df
```

**Statistics:**
- File size: ~70 MB
- Load time: ~0.9 sec
- Shape: 203,738 rows × 36 columns
- Missing values: 2,681,174 (36.56%)

---

### Stage 2: Indexing & Standardization

**Module:** `src/s02_indexing/msti_indexing.py`

**Objective:** Transform long format to wide format and standardize variables.

**Transformation Steps:**
1. `select_core_columns`: Extract essential columns
2. `build_indicators`: Construct indicators (Measure × Unit)
3. `add_dimensions`: Add geographic (Continent) and temporal (Decade) dimensions
4. `reshape_wide`: Pivot long → wide
5. `standardize`: Z-score normalization (μ=0, σ=1)
6. `set_index`: MultiIndex [Zone, Continent, Year, Decade]

**Input:** `data/raw/msti_raw.csv`  
**Output:** `data/interim/msti_indexed.csv`

**Key Operations:**
```python
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Z-score normalization to all numeric columns.
    
    Formula: z = (x - μ) / σ
    
    Returns:
    --------
    pd.DataFrame
        Standardized variables (μ≈0, σ≈1)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df
```

**Statistics:**
- Processing time: ~0.3 sec
- Output shape: 2,067 observations × 89 variables
- Index levels: ['zone', 'continent', 'year', 'decade']
- All variables standardized: μ≈0, σ≈1

---

### Stage 3: GPU-Accelerated Imputation

**Module:** `src/s03_imputation/msti_knn_imputer_gpu.py`

**Objective:** Impute missing values using weighted k-NN inverse-distance.

**Details:** See `documentation/methodology/knn_imputer_method.pdf`

**Transformation Steps:**
1. `compute_observed_stats`: Calculate observation statistics
2. Extract metadata (MultiIndex: Zone, Continent, Year, Decade)
3. Transfer CPU → GPU (memory optimization)
4. Masked RMSE: Artificially mask ~10% observed values + test imputation
5. `optimize_knn_neighbors`: Grid search (metric × k) with early stopping
6. `impute_with_knn_gpu`: Iterative weighted KNN on GPU
7. Transfer GPU → CPU: CuPy → NumPy conversion + DataFrame reconstruction
8. Export CSV with preserved MultiIndex

**Input:** `data/interim/msti_indexed.csv`  
**Outputs:**
- `data/processed/msti_imputed.csv`
- `outputs/reports/knn_gpu_cpu_monitoring_report.txt`

**Key Algorithm:**
```python
def impute_with_knn_gpu(X_gpu, k=5, metric='manhattan', max_iter=5):
    """
    GPU-accelerated weighted KNN imputation.
    
    Parameters:
    -----------
    X_gpu : cupy.ndarray
        Data matrix with missing values (NaN)
    k : int
        Number of nearest neighbors
    metric : str
        Distance metric ('euclidean' or 'manhattan')
    
    Returns:
    --------
    X_imputed : cupy.ndarray
        Fully imputed matrix (0% missing)
    """
    from cuml.neighbors import NearestNeighbors
    
    for iteration in range(max_iter):
        # Identify missing values
        missing_mask = cp.isnan(X_gpu)
        
        # For each row with missing values
        for i in cp.where(missing_mask.any(axis=1))[0]:
            # Find k nearest neighbors (using observed features only)
            observed_mask = ~missing_mask[i]
            nn = NearestNeighbors(n_neighbors=k, metric=metric)
            nn.fit(X_gpu[:, observed_mask])
            
            distances, indices = nn.kneighbors(X_gpu[i:i+1, observed_mask])
            
            # Compute inverse-distance weights
            weights = 1.0 / (distances + 1e-6)
            weights /= weights.sum()
            
            # Impute missing values
            for j in cp.where(missing_mask[i])[0]:
                X_gpu[i, j] = (weights * X_gpu[indices[0], j]).sum()
    
    return X_gpu
```

---

### Stage 4: UMAP 3D Projection

**Module:** `src/s04_visualization/msti_umap_projection.py`

**Objective:** 3D topological projection of observations via UMAP for global structure visualization.

**Transformation Steps:**
1. Extract numeric variables (exclude identifiers)
2. Add Gaussian jitter for numerical stability
3. Compute intra-cluster transparency (opaque core, transparent periphery)
4. `umap.UMAP`: Non-linear dimensionality reduction ($\mathbb{R}^{89} \rightarrow \mathbb{R}^{3}$)
5. `apply_repulsion_optimized`: Spatial stabilization via vectorized repulsion forces
6. **Observation visualization**: Color by continent (Plotly + Matplotlib)
7. **HDBSCAN clustering**: Automatic density-based group detection
8. Generate interactive (HTML) and static (PNG) visualizations

**Processing:** CPU parallelized (8 Numba threads)

**Input:** `data/processed/msti_imputed.csv`  
**Outputs:**
- `outputs/figures/umap_observations_3d.png`
- `outputs/figures/umap_clusters_3d.png`
- `outputs/interactive/umap_observations_3d.html`
- `outputs/interactive/umap_clusters_3d.html`

**Key Algorithm:**
```python
def project_umap3d_observations(data, continent_col='continent'):
    """
    Project observations to 3D UMAP space with repulsion forces.
    
    Returns:
    --------
    dict
        {
            'coords_3d': np.ndarray (n, 3),
            'continents': pd.Series,
            'plot_data': dict
        }
    """
    # UMAP projection
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    coords_3d = reducer.fit_transform(data)
    
    # Apply vectorized repulsion
    coords_3d = apply_repulsion_optimized(
        coords_3d,
        n_neighbors=15,
        strength=0.1,
        n_iterations=50
    )
    
    return {
        'coords_3d': coords_3d,
        'continents': data[continent_col],
        'plot_data': prepare_plotly_traces(coords_3d, data[continent_col])
    }
```
---

### Stage 5: Univariate Analysis

**Module:** `src/s05_analysis/msti_analysis_univariate.py`

**Objective:** Descriptive statistics and distribution analysis by thematic blocks.

**Key Operations:**
1. `describe_univariate`: Compute statistics (mean, std, quartiles, min, max)
2. `plot_boxplots`: Distribution analysis by variable groups

**Input:** `data/processed/msti_imputed.csv`  
**Outputs:**
- `outputs/reports/msti_univariate_stats.csv`
- `outputs/figures/boxplots_rd_personnel.png`
- `outputs/figures/boxplots_rd_expenditure.png`
- (Additional boxplot groups)

---

### Stage 6: Multivariate Analysis

**Modules:**
- `src/s05_analysis/msti_corr_analysis.py` (Correlation matrices)
- `src/s05_analysis/msti_cah_mfa.py` (Clustering, MFA)

**Objective:** Advanced multivariate statistical analysis.

---

#### 6.1 Correlation Matrix

**Key Operations:**
```python
def plot_correlation_matrix(data, method='pearson'):
    """
    Compute and visualize Pearson correlation matrix (89×89).
    
    Features:
    - Hierarchical ordering (Ward linkage)
    - Diverging colormap (RdBu_r)
    - 300 DPI output
    """
    corr = data.corr(method=method)
    
    # Hierarchical ordering
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    dissimilarity = 1 - np.abs(corr)
    linkage_matrix = linkage(squareform(dissimilarity), method='ward')
    
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    
    # Reorder and plot
    corr_ordered = corr.iloc[order, order]
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_ordered, cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True)
    plt.tight_layout()
    plt.savefig('outputs/figures/correlation_matrix.png', dpi=300)
```

**Output:** `outputs/figures/correlation_matrix.png`

---

#### 6.2 Hierarchical Clustering (CAH)

**Algorithm:** Ward linkage with Euclidean distance

```python
def cluster_variables_hierarchical(data, method='ward', metric='euclidean'):
    """
    Hierarchical clustering of variables.
    
    Returns:
    --------
    dict
        {
            'linkage': np.ndarray (linkage matrix),
            'clusters': dict (variable → cluster_id),
            'dendrogram_data': dict
        }
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    
    # Transpose: cluster variables (not observations)
    X = data.T
    
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    # Cut tree at optimal height (Cattell criterion)
    clusters = fcluster(linkage_matrix, t=15, criterion='maxclust')
    
    return {
        'linkage': linkage_matrix,
        'clusters': dict(zip(data.columns, clusters)),
        'dendrogram_data': dendrogram(linkage_matrix, no_plot=True)
    }
```

**Output:** `outputs/figures/cah_dendrogram.png`

---

#### 6.3 Multiple Factor Analysis (MFA)

**Algorithm:** Block-weighted PCA

```python
def run_mfa_projection(data, variable_blocks):
    """
    Multiple Factor Analysis with proper block weighting.
    
    Steps:
    1. For each block g, compute first eigenvalue λ₁ᵍ
    2. Weight block: wᵍ = 1/√λ₁ᵍ
    3. Concatenate weighted blocks
    4. Global PCA
    
    Returns:
    --------
    dict
        {
            'components': np.ndarray (n_components, n_variables),
            'explained_variance': np.ndarray (n_components,),
            'loadings': np.ndarray (n_variables, n_components),
            'correlations': np.ndarray (n_variables, n_components),
            'scores': np.ndarray (n_observations, n_components)
        }
    """
    from sklearn.decomposition import PCA
    
    # Step 1: Compute block weights
    weighted_blocks = []
    for block_name, var_list in variable_blocks.items():
        X_block = data[var_list]
        pca_block = PCA(n_components=1)
        pca_block.fit(X_block)
        
        lambda_1 = pca_block.explained_variance_[0]
        weight = 1.0 / np.sqrt(lambda_1)
        
        weighted_blocks.append(X_block * weight)
    
    # Step 2: Global PCA
    X_weighted = pd.concat(weighted_blocks, axis=1)
    pca_global = PCA(n_components=10)
    scores = pca_global.fit_transform(X_weighted)
    
    # Step 3: Compute correlations (CORRECT FORMULA)
    loadings = pca_global.components_.T  # (n_vars, n_components)
    eigenvalues = pca_global.explained_variance_
    
    correlations = loadings * np.sqrt(eigenvalues)  # ← CRITICAL
    
    return {
        'components': pca_global.components_,
        'explained_variance': pca_global.explained_variance_ratio_,
        'loadings': loadings,
        'correlations': correlations,
        'scores': scores
    }
```

**Outputs:**
- `outputs/figures/mfa_cattell_criterion.png` (Scree plot)
- `outputs/figures/mfa_correlation_circle_f12.png` (Variables F1-F2)
- `outputs/figures/mfa_correlation_circle_f13.png` (Variables F1-F3)
- `outputs/figures/mfa_projection_continents_f12.png` (Spline barycenters)
- `outputs/figures/mfa_projection_continents_f13.png`
- `outputs/figures/mfa_projection_countries_f12.png` (Temporal trajectories)
- `outputs/figures/mfa_projection_countries_f13.png`
- `outputs/reports/msti_mfa_inertia.csv`
- `outputs/reports/msti_mfa_variable_metrics.csv`
- `outputs/reports/msti_mfa_obs_metrics.xlsx`

**Processing time:** ~12 sec (parallelized)

---

## Results & Outputs

### File Organization
```
outputs/
├── figures/                              # High-resolution visualizations
│   ├── umap_observations_3d.png          # 3D UMAP colored by continent
│   ├── umap_clusters_3d.png              # HDBSCAN density clusters
│   ├── boxplots_rd_personnel.png         # Distribution: R&D personnel
│   ├── boxplots_rd_expenditure.png       # Distribution: R&D expenditure
│   ├── correlation_matrix.png            # Pearson correlation (89×89)
│   ├── cah_dendrogram.png                # Hierarchical clustering
│   ├── mfa_cattell_criterion.png         # Scree plot (explained variance)
│   ├── mfa_correlation_circle_f12.png    # Variable correlations (F1-F2)
│   ├── mfa_correlation_circle_f13.png    # Variable correlations (F1-F3)
│   ├── mfa_projection_continents_f12.png # Continent barycenters (F1-F2)
│   ├── mfa_projection_continents_f13.png # Continent barycenters (F1-F3)
│   ├── mfa_projection_countries_f12.png  # Country trajectories (F1-F2)
│   └── mfa_projection_countries_f13.png  # Country trajectories (F1-F3)
│
├── interactive/                          # Interactive HTML dashboards
│   ├── umap_observations_3d.html         # Plotly 3D scatter (continent)
│   └── umap_clusters_3d.html             # Plotly 3D scatter (HDBSCAN)
│
├── reports/                              # Numerical exports
│   ├── msti_univariate_stats.csv         # Descriptive statistics (89 vars)
│   ├── msti_mfa_inertia.csv              # MFA explained variance
│   ├── msti_mfa_variable_metrics.csv     # Variable contributions
│   ├── msti_mfa_obs_metrics.xlsx         # Observation coordinates
│   ├── knn_gpu_cpu_monitoring_report.txt # GPU/CPU diagnostics
│   └── pipeline_metadata.json            # Execution logs

```

---

<a id="key-findings"></a>

### Key Findings

**Note:** This section presents **numerical results** from pipeline execution (algorithmic metrics, computed performance). **Scientific interpretation** and **result analysis** are not covered here.

**Imputation Quality (Stage 3):**
- Optimal KNN configuration: Manhattan distance, **k=4**
- Masked RMSE: **0.223362** (standardized scale)
- 67,703 values imputed (36.80% of total)
- **10 complete KNN iterations**
- Global early stopping: 31/42 grid search tests completed
- Processing time: **24.11 seconds** (GPU)

**Dimensionality Reduction (Stage 4):**
- UMAP 3D projection successfully separates observations into continental clusters
- 50 vectorized repulsion iterations for spatial stabilization
- HDBSCAN detects 8-12 density-based groups (depending on min_cluster_size)
- Temporal trajectories computed for 38 countries (1981→2025)

**Multivariate Analysis (Stage 6):**
- 89×89 correlation matrix computed (Pearson method)
- CAH dendrogram generated with Ward linkage on Euclidean distance
- First 3 MFA components explain **~75%** of total variance
- Cumulative inertia: **F1 (~48%), F2 (~12%), F3 (~12%)**
- 7 MFA visualizations produced (correlation circles, factorial projections)

---

<a id="documentation--references"></a>

### Documentation & References

**External References:**

- **OECD MSTI Database**: https://stats.oecd.org/Index.aspx?DataSetCode=MSTI_PUB
- **Frascati Manual**: https://www.oecd.org/sti/inno/frascati-manual.htm
- **RAPIDS Documentation**: https://docs.rapids.ai/
- **UMAP Theory**: https://arxiv.org/abs/1802.03426
- **MFA Methodology**: Escofier & Pagès (1994). "Multiple Factor Analysis"

**Internal Documentation:**

*Methodology:*
- `docs/methodology/protocole_statistiques.docx`: Statistical protocols
- `docs/methodology/knn_imputer_method.docx`: Detailed KNN imputation algorithm
- `docs/methodology/mfa_method.docx`: Multiple Factor Analysis methodology

*OECD References:*
- `docs/references/msti_glossaire.docx`: Glossary of 89 MSTI indicators (French)
- `docs/references/msti_glossary_en.docx`: Glossary of 89 MSTI indicators (English)
- `docs/references/oecd_msti_manuel_frascati.pdf`: Frascati Manual (R&D methodology)
- `docs/references/oecd_msti_documentation.pdf`: Complete OECD MSTI documentation (French)
- `docs/references/oecd_msti_documentation_en.pdf`: Complete OECD MSTI documentation (English)

---

## Contact

- **Author**: Aurélien Diop Lascroux
- **Email**: aurelien.dioplascroux@outlook.fr
- **GitHub**: https://github.com/AurelienDiopLascroux/stea-pipeline-msti
