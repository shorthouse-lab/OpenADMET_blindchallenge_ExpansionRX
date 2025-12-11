# openadmet_dataset.py
"""
Custom dataset loader for OpenADMET blind challenge data.
Handles loading CSV files with SMILES and multiple target properties.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import features
from experiments import Method


class OpenADMETDataset:
    """
    Dataset loader for OpenADMET challenge data.

    Supports loading CSV files with SMILES and multiple target properties.
    Target columns are automatically detected (all numeric columns except indices).
    """

    # Available target properties in ExpansionRX dataset
    TARGET_PROPERTIES = [
        'LogD',
        'KSOL',
        'HLM CLint',
        'MLM CLint',
        'Caco-2 Permeability Papp A>B',
        'Caco-2 Permeability Efflux',
        'MPPB',
        'MBPB',
        'MGMB'
    ]

    def __init__(self, csv_path: str, target_property: str, feature_type: str,
                 smiles_column: str = 'SMILES', cache_dir: str = None, use_cache: bool = True):
        """
        Initialize OpenADMET dataset.

        Args:
            csv_path: Path to CSV file containing SMILES and target properties
            target_property: Name of the target property column to predict
            feature_type: Type of molecular features to generate (e.g., 'ECFP', 'MACCS', etc.)
            smiles_column: Name of the column containing SMILES strings
        """
        self.csv_path = csv_path
        self.target_property = target_property
        self.feature_type = feature_type
        self.smiles_column = smiles_column
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent.parent / '.cache'
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load and process data
        self._load_data()

    def _load_data(self):
        """Load data from CSV and generate features (Modified to persist SMILES)"""
        # Read CSV
        df = pd.read_csv(self.csv_path)

        # Validate columns
        if self.smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{self.smiles_column}' not found in CSV")
        if self.target_property not in df.columns:
            raise ValueError(f"Target property '{self.target_property}' not found in CSV")

        # Get raw SMILES and target values from the dataframe
        raw_smiles = df[self.smiles_column].values
        all_targets = df[self.target_property].values

        # Convert SMILES to RDKit molecules
        mols = [Chem.MolFromSmiles(s) for s in raw_smiles]

        # Attempt to load features from cache
        cache_key = f"{Path(self.csv_path).name}_{self.target_property}_{self.feature_type}.npz".replace(' ', '_')
        cache_path = self.cache_dir / cache_key

        if self.use_cache and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                self.X = cached['X']
                valid_indices = cached['valid_indices']
                print(f"Loaded cached features: {cache_path}")
            except Exception:
                self.X = None
        else:
            self.X = None

        # Generate molecular features if not cached
        if self.X is None or (isinstance(self.X, np.ndarray) and self.X.size == 0):
            print(f"Generating {self.feature_type} features for {len(mols)} molecules...")
            self.X, valid_indices = features.getFeature(mols, self.feature_type)
            # Save to cache
            try:
                if not isinstance(self.X, np.ndarray) or self.X.dtype == object:
                    pass
                else:
                    np.savez_compressed(cache_path, X=self.X, valid_indices=np.array(valid_indices))
                    print(f"Cached features to: {cache_path}")
            except Exception:
                pass

        # --- MODIFICATION START ---
        # 1. Filter Targets AND SMILES using valid_indices (successful featurization)
        # We perform this filtering before checking for NaN targets
        self.Y = all_targets[valid_indices]
        current_smiles = np.array(raw_smiles)[valid_indices]
        # --- MODIFICATION END ---

        # Handle missing values in targets
        self.Y = pd.to_numeric(self.Y, errors='coerce')

        # Store indices of non-missing values
        non_missing_mask = ~np.isnan(self.Y)

        if not non_missing_mask.any():
            raise ValueError(f"No valid (non-missing) data points found for target '{self.target_property}'")

        # Final Filter: X, Y, and SMILES must all remain aligned based on valid targets
        self.X = self.X[non_missing_mask]
        self.Y = self.Y[non_missing_mask]

        # --- MODIFICATION START ---
        self.smiles = current_smiles[non_missing_mask]
        # --- MODIFICATION END ---

        # Store original indices (from the valid molecules)
        self.original_indices = np.array(valid_indices)[non_missing_mask]

        # Store molecule names if available
        if 'Molecule Name' in df.columns:
            self.molecule_names = df['Molecule Name'].values[valid_indices][non_missing_mask]
        else:
            self.molecule_names = None

        print(f"Loaded {len(self.Y)} valid data points for {self.target_property}")
        print(f"Target statistics: mean={np.mean(self.Y):.3f}, std={np.std(self.Y):.3f}, "
              f"min={np.min(self.Y):.3f}, max={np.max(self.Y):.3f}")


    @staticmethod
    def get_available_targets(csv_path: str, smiles_column: str = 'SMILES') -> dict:
        """
        Get list of available target properties and their data availability.

        Args:
            csv_path: Path to CSV file
            smiles_column: Name of SMILES column

        Returns:
            Dictionary mapping target names to number of available (non-missing) values
        """
        df = pd.read_csv(csv_path)

        # Identify potential target columns (numeric columns, excluding index/ID columns)
        exclude_patterns = ['index', 'id', 'name', 'smiles', 'molecule']
        target_cols = []

        for col in df.columns:
            col_lower = col.lower()
            # Skip if it's SMILES or looks like an ID column
            if col == smiles_column or any(pattern in col_lower for pattern in exclude_patterns):
                continue
            # Include if it has numeric data
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == 'object':
                target_cols.append(col)

        # Count non-missing values for each target
        availability = {}
        for col in target_cols:
            non_missing = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if non_missing > 0:
                availability[col] = int(non_missing)

        return availability


def create_dataset_from_method(method: Method, csv_path: str) -> OpenADMETDataset:
    """
    Helper function to create an OpenADMETDataset from a Method object.
    Expects the dataset name to be in the format: "OpenADMET_{target_property}"

    Args:
        method: Method object with dataset, feature, and model specifications
        csv_path: Path to the CSV file containing the data

    Returns:
        OpenADMETDataset instance
    """
    # Extract target property from dataset name
    # Expected format: "OpenADMET_LogD", "OpenADMET_KSOL", etc.
    if not method.dataset.startswith("OpenADMET_"):
        raise ValueError(f"Dataset name must start with 'OpenADMET_', got: {method.dataset}")

    target_property = method.dataset.replace("OpenADMET_", "")

    # Replace underscores with spaces for multi-word properties
    # E.g., "HLM_CLint" -> "HLM CLint"
    target_property = target_property.replace("_", " ")

    return OpenADMETDataset(
        csv_path=csv_path,
        target_property=target_property,
        feature_type=method.feature
    )
