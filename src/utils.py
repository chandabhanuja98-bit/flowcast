import json
import re
from datetime import datetime, timezone

import joblib
import pandas as pd

from features import DATASET_VERSION
from ingest import PROJECT_ROOT

MODEL_DIRECTORY = PROJECT_ROOT / "models"


def model_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def model_path(name: str, target: str, extension: str):
    return MODEL_DIRECTORY / f"{target}__{model_slug(name)}.{extension}"


def save_model(model, name: str, target: str, feature_columns: list[str], train: pd.DataFrame, metrics: dict, scaler=None, class_weights=None) -> None:
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_columns": feature_columns, "scaler": scaler, "class_weights": class_weights}
    joblib.dump(bundle, model_path(name, target, "joblib"))

    card = {
        "model": name,
        "target": target,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "training_window": {"from": str(train["timestamp"].min()), "to": str(train["timestamp"].max()), "rows": len(train)},
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "hyperparameters": model.get_params() if hasattr(model, "get_params") else {},
        "class_weights": None if class_weights is None else [float(weight) for weight in class_weights],
        "metrics": metrics,
    }
    model_path(name, target, "json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")


def load_model(name: str, target: str) -> dict:
    return joblib.load(model_path(name, target, "joblib"))
