import inspect
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

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
DEFAULT_FINETUNE_CTX_PLUS_QUERY_SAMPLES = 2048
DEFAULT_NUM_ESTIMATORS_FINETUNE = 2
DEFAULT_NUM_ESTIMATORS_VALIDATION = 2
DEFAULT_NUM_ESTIMATORS_FINAL_INFERENCE = 2


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


def _build_finetune_wrapper(
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    n_ctx_plus_query: Optional[int],
    n_estimators_finetune: Optional[int],
    n_estimators_validation: Optional[int],
    n_estimators_final: Optional[int],
    logger,
) -> Tuple[Optional[object], bool]:
    """
    Try to build the FinetunedTabPFNRegressor wrapper if available.
    Returns (model, used_wrapper_flag).
    """
    try:
        from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
    except Exception as exc:
        logger.info("FinetunedTabPFNRegressor unavailable (%s); falling back to base model.", exc)
        return None, False

    sig = inspect.signature(FinetunedTabPFNRegressor.__init__)
    kwargs = {}

    def _maybe_add(candidates, value) -> None:
        for candidate in candidates if isinstance(candidates, (list, tuple)) else (candidates,):
            if candidate in sig.parameters:
                kwargs[candidate] = value
                return

    _maybe_add("device", device)
    _maybe_add(("epochs", "n_epochs", "num_epochs"), epochs)
    _maybe_add(("learning_rate", "lr"), lr)
    _maybe_add(("batch_size", "bs"), batch_size)
    _maybe_add(
        ("n_finetune_ctx_plus_query_samples", "finetune_ctx_plus_query_samples", "ctx_plus_query_samples"),
        n_ctx_plus_query,
    )
    _maybe_add("n_estimators_finetune", n_estimators_finetune)
    _maybe_add("n_estimators_validation", n_estimators_validation)
    _maybe_add("n_estimators_final_inference", n_estimators_final)

    try:
        model = FinetunedTabPFNRegressor(**kwargs)
        logger.info(
            "Using FinetunedTabPFNRegressor with args: %s",
            {k: v for k, v in kwargs.items()},
        )
        return model, True
    except Exception as exc:
        logger.warning("Could not instantiate FinetunedTabPFNRegressor: %s", exc)
        return None, False


def _safe_target_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _save_model(
    model,
    checkpoint_dir: Optional[Union[str, Path]],
    target: str,
    logger,
) -> None:
    """Persist a fine-tuned model if saving utilities are available."""
    if checkpoint_dir is None:
        return

    try:
        from tabpfn.model_loading import save_tabpfn_model
    except Exception as exc:
        logger.warning("Model saving unavailable: %s", exc)
        return

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"tabpfn_finetune_{_safe_target_name(target)}.pt"
    try:
        save_tabpfn_model(model, ckpt_path)
        logger.info("Saved fine-tuned model for %s to %s", target, ckpt_path)
    except Exception as exc:
        logger.warning("Could not save model for %s: %s", target, exc)


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
    n_ctx_plus_query: Optional[int] = DEFAULT_FINETUNE_CTX_PLUS_QUERY_SAMPLES,
    n_estimators_finetune: Optional[int] = DEFAULT_NUM_ESTIMATORS_FINETUNE,
    n_estimators_validation: Optional[int] = DEFAULT_NUM_ESTIMATORS_VALIDATION,
    n_estimators_final: Optional[int] = DEFAULT_NUM_ESTIMATORS_FINAL_INFERENCE,
    checkpoint_dir: Optional[Union[str, Path]] = None,
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

        wrapper_model, used_wrapper = _build_finetune_wrapper(
            device=device,
            epochs=finetune_epochs,
            lr=finetune_lr,
            batch_size=finetune_batch_size,
            n_ctx_plus_query=n_ctx_plus_query,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final=n_estimators_final,
            logger=logger,
        )

        model = wrapper_model or TabPFNRegressor(device=device)
        X_tr = X_train.loc[valid_mask].values
        y_tr = y_train.loc[valid_mask].values

        if wrapper_model is not None:
            output_dir = None
            if checkpoint_dir is not None:
                output_dir = Path(checkpoint_dir) / f"{_safe_target_name(target)}_runs"
                output_dir.mkdir(parents=True, exist_ok=True)
            model.fit(X_tr, y_tr, output_dir=output_dir)
        else:
            model.fit(X_tr, y_tr)
        if not used_wrapper:
            _try_finetune(
                model=model,
                X=X_tr,
                y=y_tr,
                epochs=finetune_epochs,
                lr=finetune_lr,
                batch_size=finetune_batch_size,
                logger=logger,
            )
        _save_model(model, checkpoint_dir, target, logger)
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
    FINETUNE_CTX_PLUS_QUERY = DEFAULT_FINETUNE_CTX_PLUS_QUERY_SAMPLES
    NUM_ESTIMATORS_FINETUNE = DEFAULT_NUM_ESTIMATORS_FINETUNE
    NUM_ESTIMATORS_VALIDATION = DEFAULT_NUM_ESTIMATORS_VALIDATION
    NUM_ESTIMATORS_FINAL = DEFAULT_NUM_ESTIMATORS_FINAL_INFERENCE
    CHECKPOINT_DIR = base_dir / "models"

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
            n_ctx_plus_query=FINETUNE_CTX_PLUS_QUERY,
            n_estimators_finetune=NUM_ESTIMATORS_FINETUNE,
            n_estimators_validation=NUM_ESTIMATORS_VALIDATION,
            n_estimators_final=NUM_ESTIMATORS_FINAL,
            checkpoint_dir=CHECKPOINT_DIR,
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
            n_ctx_plus_query=FINETUNE_CTX_PLUS_QUERY,
            n_estimators_finetune=NUM_ESTIMATORS_FINETUNE,
            n_estimators_validation=NUM_ESTIMATORS_VALIDATION,
            n_estimators_final=NUM_ESTIMATORS_FINAL,
            checkpoint_dir=CHECKPOINT_DIR,
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
