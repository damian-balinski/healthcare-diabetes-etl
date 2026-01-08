
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PipelineConfig:
    processed_dir: Path = Path("data/processed")
    cleaned_dir: Path = Path("data/cleaned")

    input_cleaned_csv: str = "diabetes_cleaned.csv"   
    input_feature_spec: str = "feature_spec.json"     

    output_dataset_parquet: str = "diabetes_model_ready.parquet"
    output_dataset_csv: str = "diabetes_model_ready.csv" 
    output_metadata_json: str = "metadata.json"


def load_inputs(cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Load processed dataset and feature spec produced earlier."""
    df = pd.read_csv(cfg.processed_dir / cfg.input_cleaned_csv)
    with open(cfg.processed_dir / cfg.input_feature_spec, "r", encoding="utf-8") as f:
        feature_spec = json.load(f)
    return df, feature_spec


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """The same FE logic as before"""
    df_out = df.copy()

    df_out["AgeGroup"] = pd.cut(
        df_out["Age"],
        bins=[0, 29, 39, 49, 59, 120],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
        right=True,
    ).astype(str)

    df_out["BMICategory"] = pd.cut(
        df_out["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False,
    ).astype(str)

    df_out["Glucose_BMI"] = df_out["Glucose"] * df_out["BMI"]
    df_out["Pregnancies_Age"] = df_out["Pregnancies"] / (df_out["Age"] + 1.0)

    return df_out


def validate_contract(df: pd.DataFrame, feature_spec: dict) -> None:
    """Fail fast if the dataset violates basic expectations"""
    target = feature_spec["target"]
    expected_cols = set(feature_spec["numeric_features"] + feature_spec["categorical_features"] + [target])

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    null_cols = df.columns[df.isna().any()].tolist()
    if null_cols:
        raise ValueError(f"Nulls detected in columns: {null_cols}")

    ranges = {
        "Glucose": (0, 300),
        "BloodPressure": (30, 200),
        "BMI": (10, 70),
        "Age": (0, 120),
    }
    
    for col, (lo, hi) in ranges.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
      
        if len(bad) > 0:
            print(
                f"[WARN] Range check for {col}: "
                f"{len(bad)} rows outside {lo}-{hi} (allowed)"
            )


def build_metadata(df: pd.DataFrame, cfg: PipelineConfig, feature_spec: dict) -> dict:
    """Create lightweight lineage/metadata for auditability."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "processed_csv": str(cfg.processed_dir / cfg.input_cleaned_csv),
            "feature_spec": str(cfg.processed_dir / cfg.input_feature_spec),
        },
        "outputs": {
            "parquet": str(cfg.cleaned_dir / cfg.output_dataset_parquet),
            "csv_optional": str(cfg.cleaned_dir / cfg.output_dataset_csv),
        },
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "target": feature_spec["target"],
        "feature_spec_snapshot": feature_spec,
    }


def save_outputs(df: pd.DataFrame, metadata: dict, cfg: PipelineConfig, export_csv: bool = False) -> None:
    """Persist dataset + metadata."""
    cfg.cleaned_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cfg.cleaned_dir / cfg.output_dataset_parquet
    df.to_parquet(parquet_path, index=False)

    if export_csv:
        csv_path = cfg.cleaned_dir / cfg.output_dataset_csv
        df.to_csv(csv_path, index=False)

    meta_path = cfg.cleaned_dir / cfg.output_metadata_json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run(export_csv: bool = False) -> None:
    cfg = PipelineConfig()
    df_in, feature_spec = load_inputs(cfg)

    df_out = apply_feature_engineering(df_in)
    validate_contract(df_out, feature_spec)

    metadata = build_metadata(df_out, cfg, feature_spec)
    save_outputs(df_out, metadata, cfg, export_csv=export_csv)

    print("ETL run complete.")
    print("Saved:", cfg.cleaned_dir / cfg.output_dataset_parquet)
    print("Saved:", cfg.cleaned_dir / cfg.output_metadata_json)


if __name__ == "__main__":
    run(export_csv=False)
