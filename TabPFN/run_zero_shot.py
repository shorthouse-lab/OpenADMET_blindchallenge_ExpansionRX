from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from build_features import FORWARD_MAP, REVERSE_MAP, inverse_transform
from utils import (
    _prepare_feature_matrices,
    attach_predictions_to_raw,
    get_device_type,
    get_logger,
)

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


def run_tabpfn_zero_shot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_names: Iterable[str],
    device: Optional[str] = None,
    fingerprint_bits: int = 2048,
    expand_fingerprint_bits: bool = True,
    fp_columns: Optional[Iterable[str]] = None,
) -> dict:
    """
    Run zero-shot TabPFNRegressor per endpoint and return predictions keyed by target.
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


def main():
    logger = get_logger("run_zero_shot")
    base_dir = Path(__file__).resolve().parent

    # Toggle normalization of targets (assumes you built matching normalized datasets)
    NORMALIZE_TARGETS = True

    dataset_dir = base_dir / "data/processed/RDKit_MD_MACCs_log"
    raw_test_path = base_dir / "data/raw/expansion_data_test_blinded.csv"
    results_output_path = base_dir / "data/results/RDKit_MD_MACCs_log/tabpfn_zero_shot.csv"

    train_df = pd.read_csv(dataset_dir / "tabpfn_train.csv")
    test_df = pd.read_csv(dataset_dir / "tabpfn_test_blinded.csv")
    ids = test_df["Molecule Name"]

    if NORMALIZE_TARGETS:
        log_target_names = [cfg[2] for cfg in FORWARD_MAP.values()]
        preds_log = run_tabpfn_zero_shot(
            train_df=train_df,
            test_df=test_df,
            target_names=log_target_names,
            device=get_device_type(),
            expand_fingerprint_bits=True,
        )
        pred_df_log = test_df[["SMILES", "Molecule Name"]].copy()
        for col, vals in preds_log.items():
            pred_df_log[col] = vals
        pred_df_raw = inverse_transform(pred_df_log, REVERSE_MAP)
        raw_preds = {col: pred_df_raw[col].values for col in REVERSE_MAP if col in pred_df_raw}
        attach_predictions_to_raw(
            raw_path=raw_test_path,
            predictions=raw_preds,
            output_path=results_output_path,
            ids=ids,
        )
        logger.info(f"Wrote normalized predictions to {results_output_path}")
    else:
        preds = run_tabpfn_zero_shot(
            train_df=train_df,
            test_df=test_df,
            target_names=TARGET_NAMES,
            device=get_device_type(),
            expand_fingerprint_bits=True,
        )
        attach_predictions_to_raw(
            raw_path=raw_test_path,
            predictions=preds,
            output_path=results_output_path,
            ids=ids,
        )
        logger.info(f"Wrote predictions to {results_output_path}")


if __name__ == "__main__":
    main()
