import inspect
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

DEFAULT_FINETUNE_EPOCHS = 5
DEFAULT_FINETUNE_LR = 1e-3
DEFAULT_FINETUNE_BATCH_SIZE = 256


def _try_finetune(
    model,
    X,
    y,
    epochs: int,
    lr: float,
    batch_size: int,
    logger,
) -> bool:
    """Call the best-effort fine-tune hook if the TabPFN version supports it."""
    method = None
    for name in ("finetune", "fine_tune", "finetune_on_dataset"):
        if hasattr(model, name):
            method = getattr(model, name)
            break

    if method is None:
        logger.warning("TabPFNRegressor does not expose a fine-tuning method; skipping.")
        return False

    sig = inspect.signature(method)
    kwargs = {}

    def _maybe_add(candidates, value) -> None:
        for candidate in candidates:
            if candidate in sig.parameters:
                kwargs[candidate] = value
                return

    _maybe_add(("epochs", "n_epochs", "num_epochs", "steps", "finetune_steps"), epochs)
    _maybe_add(("lr", "learning_rate", "finetune_lr"), lr)
    _maybe_add(("batch_size", "bs"), batch_size)

    try:
        method(X, y, **kwargs)
        logger.info(
            "Fine-tuned via %s%s",
            method.__name__,
            f" with args {kwargs}" if kwargs else "",
        )
        return True
    except Exception as exc:
        logger.warning("Fine-tuning failed via %s: %s", method.__name__, exc)
        return False


def run_tabpfn_finetune(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_names: Iterable[str],
    device: Optional[str] = None,
    fingerprint_bits: int = 2048,
    expand_fingerprint_bits: bool = True,
    fp_columns: Optional[Iterable[str]] = None,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    finetune_lr: float = DEFAULT_FINETUNE_LR,
    finetune_batch_size: int = DEFAULT_FINETUNE_BATCH_SIZE,
    logger=None,
) -> dict:
    """
    Run TabPFNRegressor per endpoint with an optional fine-tuning step.
    """
    device = device or get_device_type()
    logger = logger or get_logger("run_finetune")

    try:
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        raise ImportError(
            "tabpfn is not installed or failed to import. "
            "Install the tabpfn package to run fine-tuning."
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
            logger.warning("Skipping %s; target column missing.", target)
            continue

        y_train = train_df[target]
        valid_mask = y_train.notna()
        if valid_mask.sum() == 0:
            logger.warning("Skipping %s; no valid labels found.", target)
            continue

        model = TabPFNRegressor(device=device)
        X_tr = X_train.loc[valid_mask].values
        y_tr = y_train.loc[valid_mask].values

        model.fit(X_tr, y_tr)
        _try_finetune(
            model=model,
            X=X_tr,
            y=y_tr,
            epochs=finetune_epochs,
            lr=finetune_lr,
            batch_size=finetune_batch_size,
            logger=logger,
        )
        y_pred = model.predict(X_test.values)
        predictions[target] = y_pred

    return predictions


def main():
    logger = get_logger("run_finetune")
    base_dir = Path(__file__).resolve().parent

    NORMALIZE_TARGETS = True
    FINETUNE_EPOCHS = DEFAULT_FINETUNE_EPOCHS
    FINETUNE_LR = DEFAULT_FINETUNE_LR
    FINETUNE_BATCH_SIZE = DEFAULT_FINETUNE_BATCH_SIZE

    dataset_dir = base_dir / "data/processed/RDKit_MD_MACCs_log"
    raw_test_path = base_dir / "data/raw/expansion_data_test_blinded.csv"
    results_output_path = base_dir / "data/results/RDKit_MD_MACCs_log/finetune_submission.csv"

    train_df = pd.read_csv(dataset_dir / "tabpfn_train.csv")
    test_df = pd.read_csv(dataset_dir / "tabpfn_test_blinded.csv")
    ids = test_df["Molecule Name"]

    if NORMALIZE_TARGETS:
        log_target_names = [cfg[2] for cfg in FORWARD_MAP.values()]
        preds_log = run_tabpfn_finetune(
            train_df=train_df,
            test_df=test_df,
            target_names=log_target_names,
            device=get_device_type(),
            expand_fingerprint_bits=True,
            finetune_epochs=FINETUNE_EPOCHS,
            finetune_lr=FINETUNE_LR,
            finetune_batch_size=FINETUNE_BATCH_SIZE,
            logger=logger,
        )
        pred_df_log = test_df[["SMILES", "Molecule Name"]].copy()
        for col, vals in preds_log.items():
            pred_df_log[col] = vals
        pred_df_raw = inverse_transform(pred_df_log, REVERSE_MAP)
        raw_cols = [cfg[0] for cfg in REVERSE_MAP.values()]
        raw_preds = {col: pred_df_raw[col].values for col in raw_cols if col in pred_df_raw}
        attach_predictions_to_raw(
            raw_path=raw_test_path,
            predictions=raw_preds,
            output_path=results_output_path,
            ids=ids,
        )
        logger.info(f"Wrote normalized fine-tuned predictions to {results_output_path}")
    else:
        preds = run_tabpfn_finetune(
            train_df=train_df,
            test_df=test_df,
            target_names=TARGET_NAMES,
            device=get_device_type(),
            expand_fingerprint_bits=True,
            finetune_epochs=FINETUNE_EPOCHS,
            finetune_lr=FINETUNE_LR,
            finetune_batch_size=FINETUNE_BATCH_SIZE,
            logger=logger,
        )
        attach_predictions_to_raw(
            raw_path=raw_test_path,
            predictions=preds,
            output_path=results_output_path,
            ids=ids,
        )
        logger.info(f"Wrote fine-tuned predictions to {results_output_path}")


if __name__ == "__main__":
    main()
