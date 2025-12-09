# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/ingestion/msti_ingestion_load_data.py
# PURPOSE      : Load and perform exploratory overview of the OECD–MSTI raw dataset
# DESCRIPTION  :
#   - Load CSV with UTF-8 encoding and error handling
#   - Compute basic statistics (shape, missing values, dtypes)
#   - Log performance metrics (file size, load time)
#   - Validate presence of required columns
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import logging   # standardized logging
import os   # operating system interactions
import time   # performance measurement
from pathlib import Path   # path manipulations
from typing import Union   # type hints

# 1.2 --- Third-party libraries ---
import pandas as pd   # DataFrame manipulation

# 1.3 --- Internal modules ---
from src.config.msti_paths_config import DATA_PATHS   # centralized path configurations
from src.config.msti_constants import MSTI_KEY_COLUMNS_RAW   # standard MSTI key columns
from src.config.msti_system_utils import format_bytes   # utility for formatting byte sizes

# ------------------------------------------------------------------------------------------------------------
# 2. Logger Initialization
# ------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)   # create module-level logger


# ------------------------------------------------------------------------------------------------------------
# 3. Data Loading Function
# ------------------------------------------------------------------------------------------------------------
def load_raw_data(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Load the OECD–MSTI raw dataset and extract basic metadata.

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename in DATA_PATHS["raw"] directory (e.g., "msti_raw.csv").

    Returns
    -------
    pandas.DataFrame
        Raw dataset with original OECD column names.
    """
    logger.info("=== STEP: Load Raw Data ===")

    # 3.1 --- Construct file path ---
    filepath = DATA_PATHS["raw"] / filename
    filesize = os.path.getsize(filepath)

    # 3.2 --- Load CSV with timing ---
    start = time.time()
    data = pd.read_csv(filepath, encoding="utf-8")
    elapsed = time.time() - start

    # 3.3 --- Compute basic statistics ---
    n_rows, n_cols = data.shape
    n_missing = data.isna().sum().sum()
    pct_missing = (n_missing / data.size) * 100 if data.size > 0 else 0.0

    # 3.4 --- Detect key columns ---
    detected = [col for col in MSTI_KEY_COLUMNS_RAW if col in data.columns]

    # 3.5 --- Log summary ---
    logger.info(f"File: {filepath.name}")
    logger.info(f"Size: {format_bytes(filesize)} ({filesize:,} bytes)")
    logger.info(f"Load time: {elapsed:.2f}s")
    logger.info(f"Shape: {n_rows:,} rows × {n_cols} cols")
    logger.info(f"Missing: {n_missing:,} values ({pct_missing:.2f}%)")
    logger.info(f"Key columns: {len(detected)}/{len(MSTI_KEY_COLUMNS_RAW)} detected")
    logger.debug(f"Columns: {list(data.columns)}")
    logger.debug(f"Dtypes: {data.dtypes.value_counts().to_dict()}")

    return data


# ------------------------------------------------------------------------------------------------------------
# 4. Standalone Module Test
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import logging

    # 4.1 --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    print("\n" + "="*60)
    print("MSTI INGESTION - STANDALONE TEST")
    print("="*60)

    # 4.2 --- Load raw data ---
    print("\n[1/2] Loading raw dataset...")
    raw = load_raw_data("msti_raw.csv")
    print(f"    ✓ Loaded: {raw.shape[0]:,} rows × {raw.shape[1]} cols")

    # 4.3 --- Display sample ---
    print("\n[2/2] Sample data:")
    print(raw.head(5))

    print("\n" + "="*60)
    print("✓ STANDALONE TEST COMPLETE")
    print("="*60)