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

TARGET_NAMES = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]

CONVERSION_DATA = """Assay,Log_Scale,Multiplier,Log_name
LogD,False,1,LogD
KSOL,True,1e-6,LogS
HLM CLint,True,1,Log_HLM_CLint
MLM CLint,True,1,Log_MLM_CLint
Caco-2 Permeability Papp A>B,True,1e-6,Log_Caco_Papp_AB
Caco-2 Permeability Efflux,True,1,Log_Caco_ER
MPPB,True,1,Log_Mouse_PPB
MBPB,True,1,Log_Mouse_BPB
MGMB,True,1,Log_Mouse_MPB
"""


def build_conversion_maps():
    """Parse the conversion CSV string into lookup maps."""
    from io import StringIO

    conversion_df = pd.read_csv(StringIO(CONVERSION_DATA))
    conversion_df["Log_Scale"] = conversion_df["Log_Scale"].astype(bool)
    conversion_df["Multiplier"] = conversion_df["Multiplier"].astype(float)

    forward_map = {
        row["Assay"]: (row["Log_Scale"], row["Multiplier"], row["Log_name"])
        for _, row in conversion_df.iterrows()
    }
    reverse_map = {
        row["Log_name"]: (row["Assay"], row["Log_Scale"], row["Multiplier"])
        for _, row in conversion_df.iterrows()
    }
    return conversion_df, forward_map, reverse_map


CONVERSION_DF, FORWARD_MAP, REVERSE_MAP = build_conversion_maps()


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
    expand_fingerprint_bits: bool = False,
) -> dict:
    """
    Calculate a fingerprint (Morgan or MACCS). Returns a bitstring column
    by default, or one column per bit when expand_fingerprint_bits=True.
    """
    mol = _mol_from_smiles(smiles, mol_id)
    fp_type = (fp_type or "").lower()

    if fp_type == "morgan":
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=fingerprint_radius, nBits=fingerprint_bits
        )
        prefix = "morgan"
        bit_count = fingerprint_bits
    elif fp_type == "maccs":
        bit_vector = MACCSkeys.GenMACCSKeys(mol)
        prefix = "maccs"
        bit_count = bit_vector.GetNumBits()
    else:
        raise ValueError("fp_type must be 'morgan' or 'maccs'.")

    if expand_fingerprint_bits:
        bit_array = np.zeros((bit_count,), dtype=int)
        DataStructs.ConvertToNumpyArray(bit_vector, bit_array)
        return {f"{prefix}_{i}": int(bit) for i, bit in enumerate(bit_array)}

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


def forward_transform(
    train_df: pd.DataFrame,
    forward_map: Optional[dict] = None,
    include_non_log: bool = True,
) -> pd.DataFrame:
    """
    Forward-transform raw assay columns into log-space columns using the map.
    Keeps SMILES and Molecule Name; produces columns named by Log_name.
    """
    forward_map = forward_map or FORWARD_MAP
    log_train_df = train_df[["SMILES", "Molecule Name"]].copy()

    for assay, (log_scale, multiplier, log_name) in forward_map.items():
        if assay not in train_df.columns:
            continue
        x = pd.to_numeric(train_df[assay], errors="coerce")
        if log_scale:
            x = x + 1  # optional offset to avoid log(0)
            x = np.log10(x * multiplier)
            log_train_df[log_name] = x
        elif include_non_log:
            # Keep non-log targets unchanged if requested (e.g., LogD)
            log_train_df[log_name] = x
        # else skip non-log targets entirely

    expected_cols = {"SMILES", "Molecule Name"} | {v[2] for v in forward_map.values() if include_non_log or v[0]}
    missing = expected_cols - set(log_train_df.columns)
    if missing:
        raise ValueError(f"Missing transformed columns: {missing}")
    return log_train_df


def inverse_transform(
    pred_df_log: pd.DataFrame, reverse_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Reverse-transform log-space predictions back to raw assay units.
    Keeps SMILES and Molecule Name; outputs raw assay names.
    """
    reverse_map = reverse_map or REVERSE_MAP
    output_df = pred_df_log[["SMILES", "Molecule Name"]].copy()

    for log_name, (assay_name, log_scale, multiplier) in reverse_map.items():
        if log_name not in pred_df_log.columns:
            continue
        y = pd.to_numeric(pred_df_log[log_name], errors="coerce")
        if log_scale:
            y = (10 ** y) * (1 / multiplier) - 1
        output_df[assay_name] = y

    expected_cols = {"SMILES", "Molecule Name"} | {v[0] for v in reverse_map.values()}
    missing = expected_cols - set(output_df.columns)
    if missing:
        raise ValueError(f"Missing inverse-transformed columns: {missing}")
    return output_df


def run_normalization_self_checks():
    """Quick round-trip and NaN preservation checks for the conversions."""
    forward_map = FORWARD_MAP
    reverse_map = REVERSE_MAP

    synthetic = {
        "SMILES": ["C", "CC"],
        "Molecule Name": ["m1", "m2"],
    }
    for assay, (log_scale, _, _) in forward_map.items():
        synthetic[assay] = [1.5, np.nan] if log_scale else [2.5, np.nan]
    raw_df = pd.DataFrame(synthetic)

    log_df = forward_transform(raw_df, forward_map=forward_map)
    round_trip = inverse_transform(log_df, reverse_map=reverse_map)

    for assay in forward_map:
        a = raw_df[assay]
        b = round_trip[assay]
        mask = a.notna()
        if mask.any() and not np.allclose(a[mask], b[mask], equal_nan=True):
            raise AssertionError(f"Round-trip mismatch for {assay}")
        if a.isna().any() and not b.isna().equals(a.isna()):
            raise AssertionError(f"NaN preservation failed for {assay}")

    expected_log_cols = {"SMILES", "Molecule Name"} | {v[2] for v in forward_map.values()}
    if set(log_df.columns) != expected_log_cols:
        raise AssertionError("Log dataframe columns mismatch.")

    expected_raw_cols = {"SMILES", "Molecule Name"} | set(forward_map.keys())
    if set(round_trip.columns) != expected_raw_cols:
        raise AssertionError("Inverse dataframe columns mismatch.")

    return True


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

    # Remove any duplicate columns to avoid pandas reindex errors
    processed_train = processed_train.loc[:, ~processed_train.columns.duplicated()]
    processed_test = processed_test.loc[:, ~processed_test.columns.duplicated()]

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


def run_tabpfn_zero_shot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_names: Iterable[str] = TARGET_NAMES,
    device: Optional[str] = None,
    fingerprint_bits: int = 2048,
    expand_fingerprint_bits: bool = True,
    fp_columns: Optional[Iterable[str]] = None,
    additional_non_feature_cols: Optional[Iterable[str]] = None,
) -> dict:
    """
    Run zero-shot TabPFNRegressor per endpoint and return predictions keyed by target.

    Note: TabPFN does not natively support multi-task regression, so we fit/predict
    one model per target.
    """
    device = device or get_device_type()
    try:
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        raise ImportError(
            "tabpfn is not installed or failed to import. "
            "Install the tabpfn package to run zero-shot inference."
        ) from exc

    X_train, X_test, _ = _prepare_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        target_names=target_names,
        fingerprint_bits=fingerprint_bits,
        expand_fingerprint_bits=expand_fingerprint_bits,
        fp_columns=fp_columns,
        additional_non_feature_cols=additional_non_feature_cols,
    )

    predictions = {}
    for target in target_names:
        if target not in train_df.columns:
            continue

        y_train = train_df[target]
        valid_mask = y_train.notna()
        if valid_mask.sum() == 0:
            continue

        model = TabPFNRegressor(device=device)
        model.fit(X_train.loc[valid_mask].values, y_train.loc[valid_mask].values)
        y_pred = model.predict(X_test.values)
        predictions[target] = y_pred

    return predictions


def attach_predictions_to_raw(
    raw_path: Union[str, Path],
    predictions: dict,
    output_path: Union[str, Path],
    id_column: str = "Molecule Name",
    ids: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    Concatenate predictions onto the raw blinded dataframe and save to CSV.

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


def run_tabpfn_zero_shot_normalized(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: Optional[str] = None,
    fingerprint_bits: int = 2048,
    expand_fingerprint_bits: bool = True,
    fp_columns: Optional[Iterable[str]] = None,
    forward_map: Optional[dict] = None,
    reverse_map: Optional[dict] = None,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Run zero-shot TabPFN in log space (forward) and return raw-space predictions.

    Returns:
        raw_predictions: dict of assay_name -> np.ndarray (raw units)
        pred_df_log: dataframe with log-space predictions and identifiers
        pred_df_raw: dataframe with raw-space predictions and identifiers
    """
    forward_map = forward_map or FORWARD_MAP
    reverse_map = reverse_map or REVERSE_MAP

    log_forward_map = {k: v for k, v in forward_map.items() if v[0]}
    non_log_targets = [k for k, v in forward_map.items() if not v[0]]

    log_target_names = [cfg[2] for cfg in log_forward_map.values()]
    raw_predictions = {}

    if log_target_names:
        log_train_df = forward_transform(
            train_df, forward_map=log_forward_map, include_non_log=False
        )

        train_base = train_df.drop(columns=log_target_names, errors="ignore")
        train_with_logs = pd.concat(
            [train_base, log_train_df.drop(columns=["SMILES", "Molecule Name"])], axis=1
        )

        log_predictions = run_tabpfn_zero_shot(
            train_df=train_with_logs,
            test_df=test_df,
            target_names=log_target_names,
            device=device,
            fingerprint_bits=fingerprint_bits,
            expand_fingerprint_bits=expand_fingerprint_bits,
            fp_columns=fp_columns,
            additional_non_feature_cols=forward_map.keys(),  # drop raw assays from features
        )

        pred_df_log = test_df[["SMILES", "Molecule Name"]].copy()
        for col, vals in log_predictions.items():
            pred_df_log[col] = vals

        pred_df_raw = inverse_transform(pred_df_log, reverse_map=reverse_map)
        raw_predictions.update(
            {assay: pred_df_raw[assay].values for assay in log_forward_map if assay in pred_df_raw}
        )
    else:
        pred_df_log = test_df[["SMILES", "Molecule Name"]].copy()
        pred_df_raw = pred_df_log.copy()

    if non_log_targets:
        non_log_preds = run_tabpfn_zero_shot(
            train_df=train_df,
            test_df=test_df,
            target_names=non_log_targets,
            device=device,
            fingerprint_bits=fingerprint_bits,
            expand_fingerprint_bits=expand_fingerprint_bits,
            fp_columns=fp_columns,
            additional_non_feature_cols=forward_map.keys(),
        )
        raw_predictions.update(non_log_preds)
        for assay, vals in non_log_preds.items():
            pred_df_raw[assay] = vals

    return raw_predictions, pred_df_log, pred_df_raw


def run_tabpfn_finetune(*args, **kwargs):
    """
    Placeholder for a fine-tuning workflow.

    TabPFN is typically used zero-shot; fine-tuning hooks can be added here when
    a supported approach is defined (e.g., adapter training or additional epochs).
    """
    raise NotImplementedError(
        "Fine-tuning for TabPFN is not implemented yet. Use run_tabpfn_zero_shot."
    )


def build_tabpfn_input(
    raw_path: Union[str, Path],
    md_features_path: Union[str, Path],
    smiles_column: str = "SMILES",
    raw_id_column: str = "Molecule Name",
    md_id_column: str = "ID",
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    fingerprint_type: Optional[str] = "morgan",
    expand_fingerprint_bits: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load a raw dataset, merge with MD features on ID, and append RDKit 2D descriptors
    plus an optional fingerprint (Morgan or MACCS).

    By default the fingerprint is stored as a single bitstring column
    ('<fp>_fp'); set expand_fingerprint_bits=True to emit one column per bit.

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

    feature_rows = []
    for smiles, mol_id in zip(merged[smiles_column], merged["ID"]):
        row_features = gen_rdkit_desc(smiles, mol_id)
        if fingerprint_type:
            row_features.update(
                calculate_fp(
                    smiles,
                    mol_id,
                    fp_type=fingerprint_type,
                    fingerprint_radius=fingerprint_radius,
                    fingerprint_bits=fingerprint_bits,
                    expand_fingerprint_bits=expand_fingerprint_bits,
                )
            )
        feature_rows.append(row_features)

    features_df = pd.DataFrame(feature_rows)
    enriched = pd.concat([merged.reset_index(drop=True), features_df], axis=1)
    enriched = enriched.rename(columns={"ID": raw_id_column})

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_path, index=False)

    return enriched


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    train_csv_path = base_dir / "data/raw/expansion_data_train.csv"
    test_csv_path = base_dir / "data/raw/expansion_data_test_blinded.csv"

    train_MD_feat_path = base_dir / "data/features/MD_features_train.csv"
    test_MD_feat_path = base_dir / "data/features/MD_features_test_blinded.csv"

    train_output_path = base_dir / "data/processed/RDKit_MD_MACCs_log/tabpfn_train.csv"
    test_output_path = base_dir / "data/processed/RDKit_MD_MACCs_log/tabpfn_test_blinded.csv"

    results_output_path = base_dir / "data/results/RDKit_MD_MACCs_log/openadmet_submission.csv"

    # Toggle to enable log-space normalization of assay endpoints
    use_normalization = True

    if not train_output_path.exists():
        tabpfn_train = build_tabpfn_input(
            raw_path=train_csv_path,
            md_features_path=train_MD_feat_path,
            output_path=train_output_path,
            fingerprint_type="maccs",
        )

    if not test_output_path.exists():
        tabpfn_test = build_tabpfn_input(
            raw_path=test_csv_path,
            md_features_path=test_MD_feat_path,
            output_path=test_output_path,
            fingerprint_type="maccs",
        )

    train_df = pd.read_csv(train_output_path)
    test_df = pd.read_csv(test_output_path)

    ids = test_df["Molecule Name"]

    if use_normalization:
        raw_preds, pred_df_log, pred_df_raw = run_tabpfn_zero_shot_normalized(
            train_df=train_df,
            test_df=test_df,
            device=get_device_type(),
            expand_fingerprint_bits=True,  # needed for numeric features
        )
        results_df = attach_predictions_to_raw(
            raw_path=test_csv_path,
            predictions=raw_preds,
            output_path=results_output_path,
            ids=ids,
        )
    else:
        preds = run_tabpfn_zero_shot(
            train_df=train_df,
            test_df=test_df,
            target_names=TARGET_NAMES,
            device=get_device_type(),
            expand_fingerprint_bits=True,  # needed for numeric features
        )
        results_df = attach_predictions_to_raw(
            raw_path=test_csv_path,
            predictions=preds,
            output_path=results_output_path,
            ids=ids,
        )
