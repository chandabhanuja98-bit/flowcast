import copy
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluate import regression_metrics, multiclass_metrics, binary_metrics, confusion_matrix
from features import CONGESTION_ORDER, TARGET_COLUMNS
from ingest import PROJECT_ROOT
from ml_models import (
    DATASET_VERSION,
    PROCESSED_DIRECTORY,
    RANDOM_SEED,
    add_prediction_intervals,
    apply_scaler,
    calibrate_accident_risk,
    fit_scaler,
    get_feature_columns,
    load_dataset,
    merge_scoreboard,
    time_split,
    tune_class_weights,
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"
SCOREBOARD_FILE = PROJECT_ROOT / "reports" / "dl_scoreboard.csv"
REGRESSION_SCOREBOARD_FILE = PROJECT_ROOT / "reports" / "regression_scoreboard.csv"

DEFAULT_SETTINGS = {
    "cell": "gru",
    "seq": 36,
    "hidden": 96,
    "layers": 2,
    "dropout": 0.4,
    "lr": 0.0005,
    "batch": 256,
    "aux": 0.3,
    "epochs": 30,
    "patience": 5,
    "save": 1,
}
ACCIDENT_LOSS_WEIGHT = 0.2
ACCIDENT_POSITIVE_WEIGHT = 10.0
CLASSES = [0, 1, 2, 3]


def model_file(cell: str):
    return MODEL_DIRECTORY / f"{cell}_forecaster.pt"


def loss_curve_file(cell: str):
    return PROJECT_ROOT / "reports" / f"dl_loss_curve_{cell}.csv"


def predictions_file(cell: str):
    return PROCESSED_DIRECTORY / f"predictions_{cell}_{DATASET_VERSION}.csv"


class ForecastNetwork(nn.Module):
    def __init__(self, feature_count: int, settings: dict):
        super().__init__()
        recurrent = nn.GRU if settings["cell"] == "gru" else nn.LSTM
        self.recurrent = recurrent(feature_count, settings["hidden"], num_layers=settings["layers"], dropout=settings["dropout"], batch_first=True)
        self.dropout = nn.Dropout(settings["dropout"])
        self.volume_head = nn.Linear(settings["hidden"], 1)
        self.congestion_head = nn.Linear(settings["hidden"], len(CLASSES))
        self.accident_head = nn.Linear(settings["hidden"], 1)

    def forward(self, x: torch.Tensor):
        output, _ = self.recurrent(x)
        last_step = self.dropout(output[:, -1])
        return self.volume_head(last_step).squeeze(1), self.congestion_head(last_step), self.accident_head(last_step).squeeze(1)


def parse_settings(arguments: list[str]) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    for argument in arguments:
        if "=" in argument:
            key, value = argument.split("=", 1)
            settings[key] = type(DEFAULT_SETTINGS[key])(value)
        else:
            settings["cell"] = argument
    return settings


def build_sequence_data(data: pd.DataFrame, feature_columns: list[str], train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, sequence_length: int, mean=None, std=None):
    ordered = data.sort_values(["road_id", "timestamp"])
    if mean is None:
        mean, std = fit_scaler(ordered.loc[train.index, feature_columns].to_numpy(dtype=float))
    features = apply_scaler(ordered[feature_columns].to_numpy(dtype=float), mean, std).astype(np.float32)

    has_history = ordered.groupby("road_id").cumcount().to_numpy() >= sequence_length - 1
    rows = np.arange(len(ordered))
    split_rows = {
        name: rows[has_history & ordered.index.isin(part.index)]
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    }
    return ordered, torch.from_numpy(features), split_rows, mean, std


def get_batch(features: torch.Tensor, rows: np.ndarray, sequence_length: int) -> torch.Tensor:
    offsets = torch.arange(-sequence_length + 1, 1)
    return features[torch.as_tensor(rows)[:, None] + offsets]


def compute_loss(outputs, targets: dict, rows: np.ndarray, class_weights: torch.Tensor, aux: float) -> torch.Tensor:
    volume, congestion, accident = outputs
    index = torch.as_tensor(rows)
    volume_loss = nn.functional.mse_loss(volume, targets["volume"][index])
    congestion_loss = nn.functional.cross_entropy(congestion, targets["congestion"][index], weight=class_weights)
    accident_loss = nn.functional.binary_cross_entropy_with_logits(
        accident, targets["accident"][index], pos_weight=torch.tensor(ACCIDENT_POSITIVE_WEIGHT)
    )
    return volume_loss + aux * (congestion_loss + ACCIDENT_LOSS_WEIGHT * accident_loss)


def validation_loss(network: ForecastNetwork, features: torch.Tensor, targets: dict, rows: np.ndarray, class_weights: torch.Tensor, settings: dict) -> float:
    network.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), settings["batch"] * 4):
            batch_rows = rows[start:start + settings["batch"] * 4]
            outputs = network(get_batch(features, batch_rows, settings["seq"]))
            total += compute_loss(outputs, targets, batch_rows, class_weights, settings["aux"]).item() * len(batch_rows)
            count += len(batch_rows)
    return total / count


def train_network(network: ForecastNetwork, features: torch.Tensor, targets: dict, split_rows: dict, class_weights: torch.Tensor, settings: dict) -> list[dict]:
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best_loss, best_state, waited, history = float("inf"), None, 0, []

    for epoch in range(1, settings["epochs"] + 1):
        network.train()
        order = np.random.permutation(split_rows["train"])
        train_total = 0.0
        for start in range(0, len(order), settings["batch"]):
            batch_rows = order[start:start + settings["batch"]]
            optimizer.zero_grad()
            outputs = network(get_batch(features, batch_rows, settings["seq"]))
            loss = compute_loss(outputs, targets, batch_rows, class_weights, settings["aux"])
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(batch_rows)

        train_loss = train_total / len(order)
        val_loss = validation_loss(network, features, targets, split_rows["validation"], class_weights, settings)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Epoch {epoch:2d}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss, best_state, waited = val_loss, copy.deepcopy(network.state_dict()), 0
        else:
            waited += 1
            if waited >= settings["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    network.load_state_dict(best_state)
    return history


def predict(network: ForecastNetwork, features: torch.Tensor, rows: np.ndarray, settings: dict):
    network.eval()
    volumes, congestion, accidents = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), settings["batch"] * 4):
            batch = get_batch(features, rows[start:start + settings["batch"] * 4], settings["seq"])
            volume, logits, accident = network(batch)
            volumes.append(volume.numpy())
            congestion.append(torch.softmax(logits, dim=1).numpy())
            accidents.append(torch.sigmoid(accident).numpy())
    return np.concatenate(volumes), np.concatenate(congestion), np.concatenate(accidents)


def evaluate_network(network: ForecastNetwork, features: torch.Tensor, ordered: pd.DataFrame, split_rows: dict, volume_mean: float, volume_std: float, name: str, settings: dict) -> pd.DataFrame:
    truth = {
        "volume": ordered["target_volume"].to_numpy(dtype=float),
        "congestion": ordered["target_congestion"].to_numpy(dtype=int),
        "accident": ordered["target_accident"].to_numpy(dtype=int),
    }
    predictions = {split: predict(network, features, split_rows[split], settings) for split in ["validation", "test"]}
    class_weights = tune_class_weights(predictions["validation"][1], truth["congestion"][split_rows["validation"]], CLASSES)
    print(f"Class weights tuned on validation: {class_weights.round(2)}")

    rows = []
    for split in ["validation", "test"]:
        index = split_rows[split]
        volume, probabilities, accident = predictions[split]
        volume = volume * volume_std + volume_mean
        rows.append({"target": "target_volume", "model": name, "split": split, **regression_metrics(truth["volume"][index], volume)})

        for label, weights in [(name, np.ones(len(CLASSES))), (f"{name} (tuned)", class_weights)]:
            predicted = np.argmax(probabilities * weights, axis=1)
            rows.append({"target": "target_congestion", "model": label, "split": split, **multiclass_metrics(truth["congestion"][index], predicted, CLASSES)})
            if split == "test":
                print(f"Confusion matrix (test, {label}): rows = real class, columns = predicted class")
                print(confusion_matrix(truth["congestion"][index], predicted, CLASSES))

        rows.append({"target": "target_accident", "model": name, "split": split, **binary_metrics(truth["accident"][index], (accident >= 0.5).astype(int), accident)})
    return pd.DataFrame(rows)


def show_benchmark(scoreboard: pd.DataFrame, name: str) -> None:
    deep = scoreboard[(scoreboard["target"] == "target_volume") & (scoreboard["split"] == "test") & (scoreboard["model"] == name)]
    print("\nVolume - test window, deep model against classical models")
    print(deep[["model", "RMSE", "MAE", "MAPE", "R2"]].round(3).to_string(index=False))
    if REGRESSION_SCOREBOARD_FILE.exists():
        classical = pd.read_csv(REGRESSION_SCOREBOARD_FILE)
        classical = classical[(classical["target"] == "target_volume") & (classical["split"] == "test")]
        print(classical[["model", "RMSE", "MAE", "MAPE", "R2"]].round(3).to_string(index=False))

    for target in ["target_congestion", "target_accident"]:
        part = scoreboard[(scoreboard["target"] == target) & (scoreboard["split"] == "test")]
        print(f"\n{target} - test")
        print(part.drop(columns=["target", "split"]).dropna(axis=1, how="all").round(3).to_string(index=False))


def write_model_card(settings: dict, feature_columns: list[str], train: pd.DataFrame, scoreboard: pd.DataFrame, name: str, epochs_run: int) -> None:
    metrics = {}
    for _, row in scoreboard[scoreboard["model"] == name].iterrows():
        values = {key: float(row[key]) for key in row.index if key not in ("target", "model", "split") and pd.notna(row[key])}
        metrics.setdefault(row["target"], {})[row["split"]] = values

    card = {
        "model": name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "training_window": {"from": str(train["timestamp"].min()), "to": str(train["timestamp"].max()), "rows": len(train)},
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "hyperparameters": settings,
        "epochs_run": epochs_run,
        "metrics": metrics,
    }
    model_file(settings["cell"]).with_suffix(".json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")


def load_forecaster(cell: str):
    checkpoint = torch.load(model_file(cell), weights_only=False)
    if "settings" not in checkpoint:
        raise SystemExit(f"{model_file(cell).name} is from an older format. Train it again with: python3 src/dl_model.py {cell}")
    network = ForecastNetwork(len(checkpoint["feature_columns"]), checkpoint["settings"])
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network, checkpoint


def predict_all(cell: str) -> None:
    network, checkpoint = load_forecaster(cell)
    settings = checkpoint["settings"]

    data = load_dataset(PROCESSED_DIRECTORY / f"features_{DATASET_VERSION}.csv")
    train, validation, test = time_split(data)
    ordered, features, split_rows, _, _ = build_sequence_data(
        data, checkpoint["feature_columns"], train, validation, test, settings["seq"], checkpoint["mean"], checkpoint["std"]
    )

    level_names = {code: level for level, code in CONGESTION_ORDER.items()}
    frames = []
    for split in ["validation", "test"]:
        volume, probabilities, accident = predict(network, features, split_rows[split], settings)
        part = ordered.iloc[split_rows[split]].reset_index(drop=True)
        frame = pd.DataFrame({
            "road_id": part["road_id"],
            "timestamp": part["timestamp"],
            "forecast_for": part["timestamp"] + pd.Timedelta(minutes=30),
            "predicted_volume": volume * checkpoint["volume_std"] + checkpoint["volume_mean"],
            "predicted_congestion": probabilities.argmax(axis=1),
            "congestion_confidence": probabilities.max(axis=1),
            "accident_risk": accident,
        })
        frame["predicted_congestion_level"] = frame["predicted_congestion"].map(level_names)
        actual = part[TARGET_COLUMNS].rename(columns=lambda column: "actual_" + column.removeprefix("target_"))
        frames.append(pd.concat([frame, actual], axis=1).assign(split=split))

    predictions = calibrate_accident_risk(add_prediction_intervals(pd.concat(frames, ignore_index=True)))
    predictions.to_csv(predictions_file(cell), index=False)
    print(f"Saved {len(predictions)} predictions to {predictions_file(cell).name}")


def print_summary(scoreboard: pd.DataFrame, name: str, settings: dict) -> None:
    def pick(target, split, column, model=name):
        row = scoreboard[(scoreboard["target"] == target) & (scoreboard["split"] == split) & (scoreboard["model"] == model)]
        return round(float(row[column].iloc[0]), 4)

    print(
        "SUMMARY", settings,
        "| val_RMSE", pick("target_volume", "validation", "RMSE"),
        "| test_RMSE", pick("target_volume", "test", "RMSE"),
        "| test_MAPE", pick("target_volume", "test", "MAPE"),
        "| test_F1", pick("target_congestion", "test", "macro_F1"),
        "| test_AUC", pick("target_accident", "test", "ROC_AUC"),
    )


def main() -> None:
    arguments = sys.argv[1:]
    torch.set_num_threads(1)
    if arguments[:1] == ["predict"]:
        predict_all(parse_settings(arguments[1:])["cell"])
        return

    settings = parse_settings(arguments)
    name = settings["cell"].upper()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    data = load_dataset(PROCESSED_DIRECTORY / f"features_{DATASET_VERSION}.csv")
    feature_columns = get_feature_columns(data)
    train, validation, test = time_split(data)
    ordered, features, split_rows, mean, std = build_sequence_data(data, feature_columns, train, validation, test, settings["seq"])
    print(f"Sequences: train {len(split_rows['train'])}, validation {len(split_rows['validation'])}, test {len(split_rows['test'])}")

    train_volume = ordered["target_volume"].to_numpy(dtype=float)[split_rows["train"]]
    volume_mean, volume_std = float(train_volume.mean()), float(train_volume.std())
    targets = {
        "volume": torch.tensor((ordered["target_volume"].to_numpy(dtype=float) - volume_mean) / volume_std, dtype=torch.float32),
        "congestion": torch.tensor(ordered["target_congestion"].to_numpy(dtype=int), dtype=torch.long),
        "accident": torch.tensor(ordered["target_accident"].to_numpy(dtype=float), dtype=torch.float32),
    }

    counts = np.bincount(targets["congestion"][split_rows["train"]].numpy(), minlength=len(CLASSES))
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

    network = ForecastNetwork(len(feature_columns), settings)
    history = train_network(network, features, targets, split_rows, class_weights, settings)

    scoreboard = evaluate_network(network, features, ordered, split_rows, volume_mean, volume_std, name, settings)
    show_benchmark(scoreboard, name)
    print_summary(scoreboard, name, settings)

    if settings["save"]:
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": network.state_dict(), "settings": settings, "feature_columns": feature_columns, "mean": mean, "std": std,
             "volume_mean": volume_mean, "volume_std": volume_std},
            model_file(settings["cell"]),
        )
        write_model_card(settings, feature_columns, train, scoreboard, name, len(history))
        merge_scoreboard(SCOREBOARD_FILE, scoreboard)
        pd.DataFrame(history).to_csv(loss_curve_file(settings["cell"]), index=False)
        print(f"\nSaved {model_file(settings['cell']).name}, its model card, {SCOREBOARD_FILE.name} and {loss_curve_file(settings['cell']).name}")


if __name__ == "__main__":
    main()
