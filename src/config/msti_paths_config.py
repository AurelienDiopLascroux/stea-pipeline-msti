# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/utils/msti_paths_config.py
# PURPOSE      : Centralized management of directory and file paths for MSTI pipeline
# DESCRIPTION  :
#   - Auto-detects project root with fallback mechanisms
#   - Defines standardized data and output directory structure
#   - Creates directories automatically if missing
#   - Provides portable path management across environments
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import os   # operating system interfaces
from pathlib import Path   # modern OS-independent path management


# ------------------------------------------------------------------------------------------------------------
# 2. Project Root Detection
# ------------------------------------------------------------------------------------------------------------
def _detect_project_root() -> Path:
    """
    Detect project root directory with multiple fallback strategies.
    
    Detection order:
    1. Environment variable MSTI_ROOT (highest priority)
    2. Traverse up from current file until finding 'data' directory
    3. Check for 'notebooks' directory and go up one level
    
    Returns
    -------
    pathlib.Path
        Absolute path to project root.
    
    Raises
    ------
    RuntimeError
        If project root cannot be detected.
    """
    # 2.1 --- Environment variable override --- 
    if "MSTI_ROOT" in os.environ:
        root = Path(os.environ["MSTI_ROOT"]).resolve()
        if root.exists():
            return root
    
    # 2.2 --- Standard layout (pipelines/utils/msti_paths_config.py) --- 
    current_file = Path(__file__).resolve()
    root = current_file.parents[2]  # Go up 2 levels: utils → pipelines → project
    
    # 2.3 --- Search upwards for 'data' directory --- 
    while not (root / "data").exists():
        if root == root.parent:  # Reached filesystem root
            raise RuntimeError(
                "Cannot detect MSTI project root. "
                "Ensure 'data/' directory exists or set MSTI_ROOT environment variable."
            )
        root = root.parent
    
    # 2.4 --- Notebook context correction --- 
    if root.name.lower() == "notebooks":
        root = root.parent
    
    return root


PROJECT_ROOT = _detect_project_root()


# -----------------------------------------------------------------------------
# 3. Directory Structure
# -----------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_PATHS = {
    "raw": DATA_DIR / "raw",
    "interim": DATA_DIR / "interim",
    "processed": DATA_DIR / "processed",
    "mappings": DATA_DIR / "mappings",
}

OUTPUT_PATHS = {
    "figures": OUTPUT_DIR / "figures",
    "reports": OUTPUT_DIR / "reports",
}


# ------------------------------------------------------------------------------------------------------------
# 4. Directory Creation
# ------------------------------------------------------------------------------------------------------------
def _create_directories() -> None:
    """
    Create all required directories if they don't exist.
    
    Returns
    -------
    None
    """
    for path_dict in (DATA_PATHS, OUTPUT_PATHS):
        for path in path_dict.values():
            path.mkdir(parents=True, exist_ok=True)

