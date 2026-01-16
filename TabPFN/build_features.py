from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from utils import (
    calculate_fp,
    expand_fp_column,
    gen_rdkit_desc,
    get_logger,
)

# Conversion table for optional normalization
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
    """Parse conversion data into forward/reverse dictionaries."""
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


def forward_transform(train_df: pd.DataFrame, conversion_dict: Dict) -> pd.DataFrame:
    """
    Apply log scaling to assay columns. Returns a dataframe with SMILES, Molecule Name,
    and the log-space targets (columns named per conversion_dict values).
    """
    log_train_df = train_df[["SMILES", "Molecule Name"]].copy()
    for assay, (log_scale, multiplier, short_name) in conversion_dict.items():
        if assay not in train_df.columns:
            continue
        x = pd.to_numeric(train_df[assay], errors="coerce")
        if log_scale:
            x = x + 1
            x = np.log10(x * multiplier)
        log_train_df[short_name] = x
    return log_train_df


def inverse_transform(pred_df: pd.DataFrame, reverse_dict: Dict) -> pd.DataFrame:
    """
    Reverse log scaling to raw assay units. Expects SMILES and Molecule Name in pred_df.
    """
    output_df = pred_df[["SMILES", "Molecule Name"]].copy()
    for log_name, (orig_name, log_scale, multiplier) in reverse_dict.items():
        if log_name not in pred_df.columns:
            continue
        y = pred_df[log_name]
        if log_scale:
            y = (10 ** y) * (1 / multiplier) - 1
        output_df[orig_name] = y
    return output_df


def add_rdkit_and_fp_features(
    df: pd.DataFrame,
    fp_type: Optional[str],
    fingerprint_bits: int = 2048,
    expand_fp_bits: bool = False,
) -> pd.DataFrame:
    """Add RDKit 2D descriptors and optional fingerprint to a dataframe with SMILES."""
    feature_rows = []
    for smiles, mol_id in zip(df["SMILES"], df["Molecule Name"]):
        row_feats = gen_rdkit_desc(smiles, mol_id)
        if fp_type:
            row_feats.update(
                calculate_fp(
                    smiles,
                    mol_id,
                    fp_type=fp_type,
                    fingerprint_bits=fingerprint_bits,
                )
            )
        feature_rows.append(row_feats)

    features_df = pd.DataFrame(feature_rows)
    enriched = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    if expand_fp_bits and fp_type:
        fp_col = f"{fp_type.lower()}_fp"
        bit_hint = fingerprint_bits if fp_type.lower() == "morgan" else None
        enriched = expand_fp_column(enriched, bit_hint, fp_column=fp_col)

    return enriched


def merge_md_features(base_df: pd.DataFrame, md_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge MD features by ID if provided."""
    if md_df is None:
        return base_df
    df_left = base_df.rename(columns={"Molecule Name": "ID"})
    md_df = md_df.rename(columns={"ID": "ID"})
    merged = df_left.merge(md_df, how="left", on="ID")
    return merged.rename(columns={"ID": "Molecule Name"})


def build_feature_set(
    raw_df: pd.DataFrame,
    md_df: Optional[pd.DataFrame],
    normalize: bool,
    fp_type: Optional[str],
    fingerprint_bits: int = 2048,
    expand_fp_bits: bool = False,
    use_rdkit_desc: bool = True,
) -> pd.DataFrame:
    """
    Construct a feature set with optional normalization, MD merge, RDKit descriptors, and fingerprints.
    """
    base_df = forward_transform(raw_df, FORWARD_MAP) if normalize else raw_df.copy()
    merged = merge_md_features(base_df, md_df if md_df is not None else None)

    if use_rdkit_desc or fp_type:
        merged = add_rdkit_and_fp_features(
            merged,
            fp_type=fp_type,
            fingerprint_bits=fingerprint_bits,
            expand_fp_bits=expand_fp_bits,
        )

    return merged


def build_and_save_datasets(configs: Iterable[dict]):
    """Build datasets for each config definition and save to disk."""
    logger = get_logger("build_features")
    base_dir = Path(__file__).resolve().parent

    raw_train_path = base_dir / "data/raw/expansion_data_train.csv"
    raw_test_path = base_dir / "data/raw/expansion_data_test_blinded.csv"

    md_train_path = base_dir / "data/features/MD_features_train.csv"
    md_test_path = base_dir / "data/features/MD_features_test_blinded.csv"

    raw_train_df = pd.read_csv(raw_train_path)
    raw_test_df = pd.read_csv(raw_test_path)
    md_train_df = pd.read_csv(md_train_path)
    md_test_df = pd.read_csv(md_test_path)

    for cfg in configs:
        name = cfg["name"]
        normalize = cfg.get("normalize", False)
        fp_type = cfg.get("fp_type")
        expand_fp_bits = cfg.get("expand_fp_bits", False)
        fingerprint_bits = cfg.get("fingerprint_bits", 2048)
        use_md = cfg.get("use_md", True)
        use_rdkit_desc = cfg.get("use_rdkit_desc", True)

        logger.info(f"Building dataset '{name}' (normalize={normalize}, fp={fp_type}, md={use_md}, rdkit={use_rdkit_desc})")

        train_df = build_feature_set(
            raw_df=raw_train_df,
            md_df=md_train_df if use_md else None,
            normalize=normalize,
            fp_type=fp_type,
            fingerprint_bits=fingerprint_bits,
            expand_fp_bits=expand_fp_bits,
            use_rdkit_desc=use_rdkit_desc,
        )
        test_df = build_feature_set(
            raw_df=raw_test_df,
            md_df=md_test_df if use_md else None,
            normalize=normalize,
            fp_type=fp_type,
            fingerprint_bits=fingerprint_bits,
            expand_fp_bits=expand_fp_bits,
            use_rdkit_desc=use_rdkit_desc,
        )

        out_dir = base_dir / "data" / "processed" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(out_dir / "tabpfn_train.csv", index=False)
        test_df.to_csv(out_dir / "tabpfn_test_blinded.csv", index=False)

        logger.info(f"Wrote {len(train_df)} train rows and {len(test_df)} test rows to {out_dir}")


if __name__ == "__main__":
    # DEFAULT_CONFIGS = [
    #     {
    #         "name": "RDKit_MD_MACCS",
    #         "normalize": False,
    #         "fp_type": "maccs",
    #         "expand_fp_bits": False,
    #         "fingerprint_bits": 2048,
    #         "use_md": True,
    #         "use_rdkit_desc": True,
    #     },
    #     {
    #         "name": "RDKit_MD",
    #         "normalize": False,
    #         "fp_type": None,
    #         "use_md": True,
    #         "use_rdkit_desc": True,
    #     },
    #     {
    #         "name": "RDKit_MD_MACCS_log",
    #         "normalize": True,
    #         "fp_type": "maccs",
    #         "expand_fp_bits": False,
    #         "fingerprint_bits": 2048,
    #         "use_md": True,
    #         "use_rdkit_desc": True,
    #     },
    # ]

    DEFAULT_CONFIGS = [
        {
            "name": "RDKit_MD_MACCS_log",
            "normalize": True,
            "fp_type": "maccs",
            "expand_fp_bits": False,
            "fingerprint_bits": 2048,
            "use_md": True,
            "use_rdkit_desc": True,
        },
    ]

    build_and_save_datasets(DEFAULT_CONFIGS)
