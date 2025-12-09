# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/indexing/msti_indexing.py
# PURPOSE      : Define measurement variables, integrate spatio-temporal dimensions, and prepare dataset
# DESCRIPTION  :
#   - Select core MSTI columns and filter aggregates
#   - Build standardized indicator labels
#   - Add spatio-temporal dimensions (continent, decade)
#   - Reshape to wide format (indicators as columns)
#   - Standardize numeric variables (z-scores)
#   - Apply MultiIndex for hierarchical data access
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------------------------------------
# 1.1 --- Standard library ---
import json   # load indicator mapping
import logging   # standardized logging
from pathlib import Path   # path manipulations
from typing import Dict   # type hints for dictionaries

# 1.2 --- Third-party libraries ---
import pandas as pd   # dataframe manipulation
from sklearn.preprocessing import StandardScaler   # feature standardization

# 1.3 --- Internal modules ---
from src.s01_ingestion.msti_ingestion_load_data import load_raw_data
from src.config.msti_constants import (
    MSTI_COLUMN_MAPPING,
    MSTI_INDEX_LABELS,
    MSTI_ID_COLUMNS,
    get_numeric_columns
)
from src.config.msti_paths_config import DATA_PATHS
from src.config.msti_variables_mapping import COUNTRY_TO_CONTINENT

# ------------------------------------------------------------------------------------------------------------
# 2. Logger Initialization
# ------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)   # create module-level logger


# ------------------------------------------------------------------------------------------------------------
# 3. Select MSTI Core Columns
# ------------------------------------------------------------------------------------------------------------
def select_core_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Select core MSTI columns and remove aggregate zones.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw OECD dataset.

    Returns
    -------
    pandas.DataFrame
        Country-level data with standardized column names.
    """
    logger.info("=== STEP 1: Select Core Columns ===")

    # 3.1 --- Select core columns ---
    core_cols = [
        "Zone de référence",
        "Mesure",
        "Unité de mesure",
        "Type de prix",
        "TIME_PERIOD",
        "OBS_VALUE",
    ]
    core_data = raw[core_cols].copy()

    # 3.2 --- Remove aggregate zones ---
    excluded_zones = [
        "Union européenne (27 pays à partir du 01/02/2020)",
        "OCDE",
    ]
    country_data = core_data[
        ~core_data["Zone de référence"].isin(excluded_zones)
    ].copy()

    # 3.3 --- Standardize column names ---
    country_data = country_data.rename(columns=MSTI_COLUMN_MAPPING)
    country_data = country_data.rename(columns={
        "TIME_PERIOD": "year",
        "OBS_VALUE": "value"
    })

    logger.info(
        f"Selected {len(country_data):,} observations "
        f"from {country_data['zone'].nunique()} countries"
    )
    return country_data


# ------------------------------------------------------------------------------------------------------------
# 4. Build Indicator Labels
# ------------------------------------------------------------------------------------------------------------
def build_indicators(countries: pd.DataFrame, mapping_file: Path) -> pd.DataFrame:
    """
    Build standardized indicator labels from measure/unit combinations.
    
    Parameters
    ----------
    countries : pandas.DataFrame
        Country-level data with measure and unit columns.
    mapping_file : pathlib.Path
        JSON file mapping raw labels to standardized names.
    
    Returns
    -------
    pandas.DataFrame
        Data with standardized 'indicator' column.
    """
    logger.info("=== STEP 2: Build Indicator Labels ===")

    # 4.1 --- Create raw labels ---
    countries["indicator"] = (
        countries["measure"].astype(str) + "/" + 
        countries["unit"].astype(str)
    )   # combine measure and unit for descriptive label

    # 4.2 --- Apply mapping ---
    with open(mapping_file, encoding="utf-8") as f:
        mapping = json.load(f)   # load standardization mapping
    countries["indicator"] = countries["indicator"].map(mapping).fillna(countries["indicator"])
        
    n_indicators = countries["indicator"].nunique()
    logger.info(f"Built {n_indicators} unique indicators")
    return countries


# ------------------------------------------------------------------------------------------------------------
# 5. Add Spatio-Temporal Dimensions
# ------------------------------------------------------------------------------------------------------------
def add_dimensions(
    country_data: pd.DataFrame, 
    geo_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Add spatio-temporal dimensions and filter to constant prices.
    
    Parameters
    ----------
    country_data : pandas.DataFrame
        Country-level data with year column.
    geo_mapping : dict
        Mapping of country names to continents.
    
    Returns
    -------
    pandas.DataFrame
        Data with continent and decade columns, constant prices only.
    """
    logger.info("=== STEP 3: Add Spatio-Temporal Dimensions ===")

    # 5.1 --- Add spatio-temporal dimensions ---
    country_data["decade"] = (country_data["year"] // 10) * 10
    country_data["continent"] = country_data["zone"].map(geo_mapping)

    # 5.2 --- Filter to constant prices ---
    is_constant = country_data["price_type"].isin([
        "N'est pas applicable", 
        "Prix constants"
    ])
    constant_prices = country_data[is_constant].copy()

    logger.info(
        f"Filtered to {len(constant_prices):,} constant-price observations"
    )
    return constant_prices


# ------------------------------------------------------------------------------------------------------------
# 6. Reshape to Wide Format
# ------------------------------------------------------------------------------------------------------------
def reshape_wide(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot dataset into wide format with indicators as columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format dataset with spatio-temporal dimensions.

    Returns
    -------
    pandas.DataFrame
        Wide-format data (observations × indicators).
    """
    logger.info("=== STEP 4: Reshape to Wide Format ===")

    # 6.1 --- Pivot to wide format ---
    wide = data.pivot_table(
        index=MSTI_INDEX_LABELS,
        columns="indicator",
        values="value"
    ).reset_index()   # reshape from long to wide format

    logger.info(f"Reshaped: {wide.shape[0]:,} rows × {wide.shape[1]} cols")
    return wide


# ------------------------------------------------------------------------------------------------------------
# 7. Standardize Features
# ------------------------------------------------------------------------------------------------------------
def standardize(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric variables to z-scores.
    
    Parameters
    ----------
    wide : pandas.DataFrame
        Wide-format data with mixed columns.
    
    Returns
    -------
    pandas.DataFrame
        Data with standardized numeric columns.
    """
    logger.info("=== STEP 5: Standardize ===")

    # 7.1 --- Identify numeric columns ---
    numeric = get_numeric_columns(wide)

    # 7.2 --- Apply z-score standardization ---
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric)

    # 7.3 --- Rebuild DataFrame ---
    scaled = pd.concat([
        wide[MSTI_ID_COLUMNS].reset_index(drop=True),
        pd.DataFrame(scaled_values, columns=numeric.columns)
    ], axis=1)

    logger.info(f"Standardized {len(numeric.columns)} variables")
    return scaled


# ------------------------------------------------------------------------------------------------------------
# 8. Set Standard MultiIndex
# ------------------------------------------------------------------------------------------------------------
def set_index(scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard MSTI MultiIndex and export to CSV.
    
    Parameters
    ----------
    scaled : pandas.DataFrame
        Standardized data with identifier columns.
    
    Returns
    -------
    pandas.DataFrame
        Data with MultiIndex applied.
    """
    logger.info("=== STEP 6: Set Standard MultiIndex ===")

    # 8.1 --- Set MultiIndex ---
    indexed = scaled.set_index(MSTI_INDEX_LABELS)

    # 8.2 --- Export to CSV ---
    output_path = DATA_PATHS["interim"] / "msti_indexed.csv"
    indexed.to_csv(output_path, sep=";", encoding="utf-8")
    logger.info(f"Indexed dataset exported to: {output_path}")
    logger.info(f"Applied MultiIndex: {indexed.index.names}")

    return indexed


# ------------------------------------------------------------------------------------------------------------
# 9. Standalone Module Test
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # 9.1 --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    print("\n" + "="*60)
    print("MSTI INDEXING - STANDALONE TEST")
    print("="*60)

    # 9.2 --- Load raw data ---
    print("\n[1/6] Loading raw dataset...")
    raw = load_raw_data("msti_raw.csv")
    print(f"    ✓ Loaded: {raw.shape[0]:,} rows")

    # 9.3 --- Execute pipeline ---
    print("\n[2/6] Executing indexing pipeline...")
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

    # 9.4 --- Display summary ---
    print("\n[3/6] Pipeline results:")
    print(f"    • Shape: {indexed.shape}")
    print(f"    • Index: {indexed.index.names}")
    print(f"    • Variables: {len(indexed.columns)}")
    
    output_path = DATA_PATHS["interim"] / "msti_indexed.csv"
    print(f"    • Saved to: {output_path}")

    print("\n[4/6] Sample data:")
    print(indexed.head(3).to_string())

    print("\n" + "="*60)
    print("✓ STANDALONE TEST COMPLETE")
    print("="*60)