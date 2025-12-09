# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/utils/msti_constants.py
# PURPOSE      : Centralized configuration constants for MSTI pipeline
# DESCRIPTION  :
#   - Defines default random state for reproducibility.
#   - Stores KNN imputation hyperparameters.
#   - Defines UMAP and repulsion algorithm parameters.
#   - Specifies column names, index structure, and data type definitions.
#   - Ensures consistent variable identification across all pipeline modules.
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
import pandas as pd   # dataframe manipulation


# ------------------------------------------------------------------------------------------------------------
# 2. Random State
# ------------------------------------------------------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42   # global seed for reproducibility


# ------------------------------------------------------------------------------------------------------------
# 3. Column and Index Structure
# ------------------------------------------------------------------------------------------------------------
# 3.1 --- Raw OECD MSTI column names ---
MSTI_KEY_COLUMNS_RAW = [
    "Zone de référence",
    "Mesure",
    "Unité de mesure",
    "Type de prix",
    "TIME_PERIOD",
    "OBS_VALUE"
]   # minimum required fields in raw OECD CSV

# 3.2 --- Standard column name mappings ---
MSTI_COLUMN_MAPPING = {
    "Zone de référence": "zone",
    "Mesure": "measure",
    "Unité de mesure": "unit",
    "Type de prix": "price_type"
}   # map French OECD names to English standardized names

# 3.3 --- Identifier columns (non-numeric metadata) ---
MSTI_ID_COLUMNS = [
    "zone",
    "continent",
    "year",
    "decade"
]   # spatio-temporal identifiers that should NOT be standardized or imputed

# 3.4 --- Standard MultiIndex structure ---
MSTI_INDEX_LABELS = [
    "zone",
    "continent",
    "year",
    "decade"
]   # hierarchical index for consistent data access across modules


# ------------------------------------------------------------------------------------------------------------
# 4. Hyperparameters
# ------------------------------------------------------------------------------------------------------------
# 4.1 --- KNN Imputation Parameters ---
KNN_IMPUTATION_PARAMS = {
    "k_list": [2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50],   # candidate neighbor counts
    "metrics": ["euclidean", "manhattan", "minkowski"], # metrics space
    "n_iter": 10,   # iterative refinement passes
    "obs_frac": 0.08,   # fraction of values to mask for validation
    "obs_max_masked": 15000,   # maximum masked cells per run
    "n_repeats": 10,   # RMSE repetitions for robustness
    "patience": 4,   # early stopping patience
    "min_improvement": 1e-4,   # minimum RMSE improvement threshold
    "batch_rmse": 4   # parallel RMSE evals per GPU batch 
}   # hyperparameters for GPU-accelerated KNN imputation

# 4.2 --- UMAP Projection Parameters ---
UMAP_DEFAULT_PARAMS = {
    "n_components": 3,   # 3D projection
    "n_neighbors": 15,   # balance local/global structure
    "min_dist": 0.1,   # moderate point spacing
    "metric": "euclidean"   # standard distance metric
}   # default UMAP configuration for dimensionality reduction

# 4.3 --- Repulsion Algorithm Parameters ---
REPULSION_DEFAULT_PARAMS = {
    "n_neighbors": 20,   # neighbors for repulsion forces
    "repulsion_strength": 0.3,   # force magnitude
    "n_iterations": 50,   # optimization steps
    "step_size": 0.05,   # gradient descent step size
    "convergence_threshold": 1e-4,   # early stopping criterion
    "rebuild_tree_every": 5   # rebuild KD-Tree every N iterations   
}   # parameters for vectorized repulsion post-processing


# ------------------------------------------------------------------------------------------------------------
# 5. Data Type Definitions
# ------------------------------------------------------------------------------------------------------------
def get_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric columns excluding identifier columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with mixed column types.
    
    Returns
    -------
    pandas.DataFrame
        Clean numeric DataFrame containing only numeric columns 
        (excluding MSTI_ID_COLUMNS).
    """
    # 5.1 --- Select numeric columns ---
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # 5.2 --- Exclude identifier columns ---    
    numeric_cols = [col for col in numeric_cols if col not in MSTI_ID_COLUMNS]

    # 5.3 --- Extract numeric block --- 
    numeric = df[numeric_cols].copy()
    
    return numeric


def validate_id_columns(df: pd.DataFrame, strict: bool = True) -> bool:
    """
    Validate presence of required identifier columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    strict : bool, default=True
        If True, raise ValueError on missing columns.
        If False, return False silently.
    
    Returns
    -------
    bool
        True if all identifier columns present.
    
    Raises
    ------
    ValueError
        If strict=True and required columns are missing.
    """
    missing = set(MSTI_ID_COLUMNS) - set(df.columns)
    
    if missing:
        if strict:
            raise ValueError(f"Missing required columns: {missing}")
        return False
    
    return True