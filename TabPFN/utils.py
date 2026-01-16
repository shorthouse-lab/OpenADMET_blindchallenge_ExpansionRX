import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

try:
    import torch
except ImportError:
    torch = None


def get_logger(name: str = "tabpfn") -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_device_type() -> str:
    """Get the device type to use for TabPFN training and inference."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _mol_from_smiles(smiles: str, mol_id: str) -> Chem.Mol:
    """Parse a SMILES into an RDKit Mol with clear errors."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"No SMILES provided for ID '{mol_id}'.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for ID '{mol_id}': {smiles!r}")
    return mol


def gen_rdkit_desc(smiles: str, mol_id: str) -> dict:
    """Calculate RDKit 2D descriptors only."""
    mol = _mol_from_smiles(smiles, mol_id)
    desc_values = _DESC_CALCULATOR.CalcDescriptors(mol)
    return dict(zip(_DESC_NAMES, desc_values))


def calculate_fp(
    smiles: str,
    mol_id: str,
    fp_type: str = "morgan",
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
) -> dict:
    """
    Calculate a fingerprint (Morgan or MACCS) and return it as a bitstring column.
    """
    mol = _mol_from_smiles(smiles, mol_id)
    fp_type = (fp_type or "").lower()

    if fp_type == "morgan":
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=fingerprint_radius, nBits=fingerprint_bits
        )
        prefix = "morgan"
    elif fp_type == "maccs":
        bit_vector = MACCSkeys.GenMACCSKeys(mol)
        prefix = "maccs"
    else:
        raise ValueError("fp_type must be 'morgan' or 'maccs'.")

    return {f"{prefix}_fp": DataStructs.BitVectToText(bit_vector)}


def expand_fp_column(
    df: pd.DataFrame,
    fingerprint_bits: Optional[int],
    fp_column: str = "morgan_fp",
    fp_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Expand a fingerprint bitstring column into one column per bit when present."""
    if fp_column not in df.columns:
        return df

    series = df[fp_column].fillna("")
    if fingerprint_bits is None:
        sample = next((s for s in series if isinstance(s, str) and s.strip()), "")
        fingerprint_bits = len(sample.strip())

    prefix = fp_prefix or fp_column.replace("_fp", "")

    def _to_bits(bitstring: Union[str, float, int]) -> list:
        if not isinstance(bitstring, str):
            bitstring = ""
        bitstring = bitstring.strip()
        bits = [1 if ch == "1" else 0 for ch in bitstring][:fingerprint_bits]
        if len(bits) < fingerprint_bits:
            bits.extend([0] * (fingerprint_bits - len(bits)))
        return bits

    bit_arrays = series.map(_to_bits)
    bit_df = pd.DataFrame(
        bit_arrays.tolist(),
        index=df.index,
        columns=[f"{prefix}_{i}" for i in range(fingerprint_bits)],
    )
    return df.drop(columns=[fp_column]).join(bit_df)


def _prepare_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_names: Iterable[str],
    fingerprint_bits: int,
    expand_fingerprint_bits: bool,
    fp_columns: Optional[Iterable[str]] = None,
    additional_non_feature_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return aligned feature matrices for train/test and training medians for imputation."""
    processed_train = train_df.copy()
    processed_test = test_df.copy()

    if expand_fingerprint_bits:
        fp_columns = list(fp_columns) if fp_columns is not None else ["morgan_fp", "maccs_fp"]
        for fp_col in fp_columns:
            bit_hint = fingerprint_bits if "morgan" in fp_col else None
            processed_train = expand_fp_column(processed_train, bit_hint, fp_column=fp_col)
            processed_test = expand_fp_column(processed_test, bit_hint, fp_column=fp_col)

    non_feature_cols = {"Molecule Name", "SMILES", *target_names}
    if additional_non_feature_cols:
        non_feature_cols |= set(additional_non_feature_cols)

    train_feature_cols = [c for c in processed_train.columns if c not in non_feature_cols]
    test_feature_cols = [c for c in processed_test.columns if c not in non_feature_cols]
    feature_cols = sorted(set(train_feature_cols) | set(test_feature_cols))

    X_train = processed_train.reindex(columns=feature_cols)
    X_test = processed_test.reindex(columns=feature_cols)

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test, medians


def attach_predictions_to_raw(
    raw_path: Union[str, Path],
    predictions: dict,
    output_path: Union[str, Path],
    id_column: str = "Molecule Name",
    ids: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    Concatenate predictions onto the raw dataframe and save to CSV.

    If ids are provided, predictions are aligned on that column; otherwise row
    order is assumed to match the dataframe used for prediction.
    """
    raw_df = pd.read_csv(raw_path)
    pred_df = pd.DataFrame(predictions)

    if ids is not None:
        pred_df[id_column] = list(ids)
        merged = raw_df.merge(pred_df, how="left", on=id_column)
    else:
        if len(pred_df) != len(raw_df):
            raise ValueError(
                f"Prediction length ({len(pred_df)}) does not match raw rows ({len(raw_df)})."
            )
        pred_df.index = raw_df.index
        merged = raw_df.copy()
        for col in pred_df.columns:
            merged[col] = pred_df[col]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged
