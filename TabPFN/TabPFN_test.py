from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def _descriptor_calculator() -> Tuple[MolecularDescriptorCalculator, Tuple[str, ...]]:
    """Return a calculator limited to RDKit 2D descriptors."""
    names: Iterable[str] = [
        name
        for name, func in Descriptors.descList
        if "Descriptors3D" not in getattr(func, "__module__", "")
    ]
    names = tuple(names)
    return MolecularDescriptorCalculator(list(names)), names


_DESC_CALCULATOR, _DESC_NAMES = _descriptor_calculator()


def _featurize_smiles(
    smiles: str, fingerprint_radius: int, fingerprint_bits: int, mol_id: str
) -> dict:
    """Calculate 2D descriptors and a Morgan fingerprint for a single SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"No SMILES provided for ID '{mol_id}'.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for ID '{mol_id}': {smiles!r}")

    desc_values = _DESC_CALCULATOR.CalcDescriptors(mol)
    descriptors = dict(zip(_DESC_NAMES, desc_values))

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=fingerprint_radius, nBits=fingerprint_bits
    )
    bit_array = np.zeros((fingerprint_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(bit_vector, bit_array)
    descriptors.update({f"morgan_{i}": int(bit) for i, bit in enumerate(bit_array)})
    return descriptors


def build_tabpfn_input(
    raw_path: Union[str, Path],
    md_features_path: Union[str, Path],
    smiles_column: str = "SMILES",
    raw_id_column: str = "Molecule Name",
    md_id_column: str = "ID",
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load a raw dataset, merge with MD features on ID, and append RDKit 2D descriptors
    plus a Morgan fingerprint.

    Returns the enriched dataframe; if output_path is provided it is also written to CSV.
    """
    raw_df = pd.read_csv(raw_path)
    md_df = pd.read_csv(md_features_path)

    if smiles_column not in raw_df.columns:
        raise KeyError(f"SMILES column '{smiles_column}' not found in {raw_path}.")
    if raw_id_column not in raw_df.columns:
        raise KeyError(f"ID column '{raw_id_column}' not found in {raw_path}.")
    if md_id_column not in md_df.columns:
        raise KeyError(f"ID column '{md_id_column}' not found in {md_features_path}.")

    raw_df = raw_df.rename(columns={raw_id_column: "ID"})
    md_df = md_df.rename(columns={md_id_column: "ID"})

    merged = raw_df.merge(md_df, how="left", on="ID")

    feature_rows = [
        _featurize_smiles(smiles, fingerprint_radius, fingerprint_bits, mol_id)
        for smiles, mol_id in zip(merged[smiles_column], merged["ID"])
    ]
    features_df = pd.DataFrame(feature_rows)
    enriched = pd.concat([merged.reset_index(drop=True), features_df], axis=1)
    enriched = enriched.rename(columns={"ID": raw_id_column})

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_path, index=False)

    return enriched

train_csv_path = Path("data/raw/expansion_data_train.csv")
test_csv_path = Path("data/raw/expansion_data_test_blinded.csv")

train_MD_feat_path = Path("data/features/MD_features_train.csv")
test_MD_feat_path = Path("data/features/MD_features_test_blinded.csv")

train_output_path = Path("data/processed/tabpfn_train.csv")
test_output_path = Path("data/processed/tabpfn_test_blinded.csv")

tabpfn_train = build_tabpfn_input(
    raw_path=train_csv_path,
    md_features_path=train_MD_feat_path,
    output_path=train_output_path
    )

