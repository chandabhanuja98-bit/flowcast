import pandas as pd
from ingest import TIMEZONE,PROJECT_ROOT
import numpy as np
from features import TARGET_COLUMNS,DATASET_VERSION,PROCESSED_DIRECTORY,CONGESTION_ORDER
from sklearn.linear_model import LinearRegression
from evaluate import regression_metrics, multiclass_metrics, confusion_matrix, binary_metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor, export_text, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor, XGBClassifier
import sys 
from sklearn.svm import SVC
from utils import save_model, load_model, model_path


SCOREBOARD_FILE = PROJECT_ROOT / "reports" / "regression_scoreboard.csv"

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
REGRESSION_TARGETS = ["target_volume", "target_travel_time"]
RANDOM_SEED = 42

CLASSIFICATION_TARGETS = ["target_congestion", "target_accident"]
CLASSIFICATION_FILE = PROJECT_ROOT / "reports" / "classification_scoreboard.csv"
SVM_SAMPLE_SIZE = 20000
ENSEMBLE_NAMES = {"forest": "Random Forest", "xgboost": "XGBoost"}
MODEL_OPTIONS = {
    "regression": ["linear", "tree", "forest", "xgboost"],
    "classification": ["svm", "tree", "forest", "xgboost"],
}
PREDICTIONS_FILE = PROCESSED_DIRECTORY / f"predictions_{DATASET_VERSION}.csv"
PRIMARY_METRICS = {
    "target_volume": ("RMSE", False),
    "target_travel_time": ("RMSE", False),
    "target_congestion": ("macro_F1", True),
    "target_accident": ("ROC_AUC", True),
}
CONFIDENCE_LEVEL = 0.90

def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"],utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values(["timestamp","road_id"]).reset_index(drop=True)

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include="number").columns
    return [column for column in numeric_columns if column not in TARGET_COLUMNS]

def time_split(df: pd.DataFrame, train_share: float = TRAIN_SHARE, validation_share: float = VALIDATION_SHARE):
    times = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    train_end = times[int(len(times) * train_share)]
    validation_end = times[int(len(times) * (train_share + validation_share))]

    train = df[df["timestamp"] < train_end]
    validation = df[(df["timestamp"] >= train_end) & (df["timestamp"] < validation_end)]
    test = df[df["timestamp"] >= validation_end]

    for name, part in [("Train", train), ("Validation", validation), ("Test", test)]:
        print(f"{name}: {len(part)} rows, {part['timestamp'].min():%Y-%m-%d %H:%M} to {part['timestamp'].max():%Y-%m-%d %H:%M}")
    return train, validation, test

class LinearRegressionScratch:

    def __init__(self, learning_rate: float = 0.05, epochs: int = 8000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.y_mean = y.mean()
        self.y_std = y.std()
        target = (y - self.y_mean) / self.y_std

        rows, columns = x.shape
        self.weights = np.zeros(columns)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            predictions = x @ self.weights + self.bias
            errors = predictions - target
            self.losses.append(float(np.mean(errors ** 2)))
            self.weights -= self.learning_rate * (2 / rows) * (x.T @ errors)
            self.bias -= self.learning_rate * 2 * errors.mean()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.weights + self.bias) * self.y_std + self.y_mean
    
def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std

def save_fitted(model, name: str, target: str, feature_columns: list[str], train: pd.DataFrame, rows: list[dict], scaler=None, class_weights=None) -> None:
    metrics = {}
    for row in rows:
        if row["target"] == target and row["model"] == name:
            metrics[row["split"]] = {key: value for key, value in row.items() if key not in ("target", "model", "split") and pd.notna(value)}
    save_model(model, name, target, feature_columns, train, metrics, scaler, class_weights)


def run_linear_regression(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    raw = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    mean, std = fit_scaler(raw["train"])
    x = {name: apply_scaler(values, mean, std) for name, values in raw.items()}

    rows = []
    for target in REGRESSION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=float),
            "validation": validation[target].to_numpy(dtype=float),
            "test": test[target].to_numpy(dtype=float),
        }
        print(f"\n===== {target} =====")
        models = [
            ("Linear Regression (scratch)", LinearRegressionScratch()),
            ("Linear Regression", LinearRegression()),
        ]
        for name, model in models:
            model.fit(x["train"], y["train"])
            if isinstance(model, LinearRegressionScratch):
                losses = model.losses
                print(f"{name} loss: epoch 1 = {losses[0]:.4f}, epoch 100 = {losses[99]:.4f}, epoch {len(losses)} = {losses[-1]:.4f}")
            for split in ["validation", "test"]:
                metrics = regression_metrics(y[split], model.predict(x[split]))
                rows.append({"target": target, "model": name, "split": split, **metrics})
            if name == "Linear Regression":
                save_fitted(model, name, target, feature_columns, train, rows, scaler=(mean, std))
    return pd.DataFrame(rows)

def run_decision_tree(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    grid = {"max_depth": [6, 10, 14], "min_samples_leaf": [5, 20, 50]}

    rows = []
    for target in REGRESSION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=float),
            "validation": validation[target].to_numpy(dtype=float),
            "test": test[target].to_numpy(dtype=float),
        }
        print(f"\n===== Decision Tree: {target} =====")
        search = GridSearchCV(
            DecisionTreeRegressor(random_state=RANDOM_SEED),
            grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_root_mean_squared_error",
        )
        search.fit(x["train"], y["train"])
        model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        for split in ["validation", "test"]:
            metrics = regression_metrics(y[split], model.predict(x[split]))
            rows.append({"target": target, "model": "Decision Tree", "split": split, **metrics})
        save_fitted(model, "Decision Tree", target, feature_columns, train, rows)

        if target == "target_volume":
            print("\nTop rules of the tree:")
            print(export_text(model, feature_names=feature_columns, max_depth=3))
    return pd.DataFrame(rows)

def run_random_forest(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    grid = {"max_depth": [10, 16], "min_samples_leaf": [5, 20]}

    rows = []
    for target in REGRESSION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=float),
            "validation": validation[target].to_numpy(dtype=float),
            "test": test[target].to_numpy(dtype=float),
        }
        print(f"\n===== Random Forest: {target} =====")
        search = GridSearchCV(
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
            grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_root_mean_squared_error",
        )
        search.fit(x["train"], y["train"])
        model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        for split in ["validation", "test"]:
            metrics = regression_metrics(y[split], model.predict(x[split]))
            rows.append({"target": target, "model": "Random Forest", "split": split, **metrics})
        save_fitted(model, "Random Forest", target, feature_columns, train, rows)

        importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
        print(f"\nTop 10 features for {target}:")
        print(importance.head(10).round(4).to_string())
    return pd.DataFrame(rows)

def run_xgboost(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    grid = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]}

    rows = []
    for target in REGRESSION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=float),
            "validation": validation[target].to_numpy(dtype=float),
            "test": test[target].to_numpy(dtype=float),
        }
        print(f"\n===== XGBoost: {target} =====")
        search = GridSearchCV(
            XGBRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
            grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_root_mean_squared_error",
        )
        search.fit(x["train"], y["train"])
        model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        for split in ["validation", "test"]:
            metrics = regression_metrics(y[split], model.predict(x[split]))
            rows.append({"target": target, "model": "XGBoost", "split": split, **metrics})
        save_fitted(model, "XGBoost", target, feature_columns, train, rows)

        importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
        print(f"\nTop 10 features for {target}:")
        print(importance.head(10).round(4).to_string())
    return pd.DataFrame(rows)

def run_svm(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    raw = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    mean, std = fit_scaler(raw["train"])
    x = {name: apply_scaler(values, mean, std) for name, values in raw.items()}
    sample = np.random.default_rng(RANDOM_SEED).choice(len(x["train"]), SVM_SAMPLE_SIZE, replace=False)

    rows = []
    for target in CLASSIFICATION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=int),
            "validation": validation[target].to_numpy(dtype=int),
            "test": test[target].to_numpy(dtype=int),
        }
        print(f"\n===== SVM: {target} =====")
        model = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)
        model.fit(x["train"][sample], y["train"][sample])
        for split in ["validation", "test"]:
            predicted = model.predict(x[split])
            if target == "target_congestion":
                metrics = multiclass_metrics(y[split], predicted, [0, 1, 2, 3])
                if split == "test":
                    print("Confusion matrix (test): rows = real class, columns = predicted class")
                    print(confusion_matrix(y[split], predicted, [0, 1, 2, 3]))
            else:
                metrics = binary_metrics(y[split], predicted, model.decision_function(x[split]))
            rows.append({"target": target, "model": "SVM", "split": split, **metrics})
        save_fitted(model, "SVM", target, feature_columns, train, rows, scaler=(mean, std))
    return pd.DataFrame(rows)

def show_classification_scoreboard(scoreboard: pd.DataFrame) -> None:
    for target in CLASSIFICATION_TARGETS:
        for split in ["validation", "test"]:
            part = scoreboard[(scoreboard["target"] == target) & (scoreboard["split"] == split)]
            part = part.drop(columns=["target", "split"]).dropna(axis=1, how="all")
            print(f"\n{target} - {split}")
            print(part.round(3).to_string(index=False))

def run_decision_tree_classifier(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    grid = {"max_depth": [6, 10, 14], "min_samples_leaf": [5, 20, 50]}
    scoring = {"target_congestion": "f1_macro", "target_accident": "roc_auc"}

    rows = []
    for target in CLASSIFICATION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=int),
            "validation": validation[target].to_numpy(dtype=int),
            "test": test[target].to_numpy(dtype=int),
        }
        print(f"\n===== Decision Tree classifier: {target} =====")
        search = GridSearchCV(
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
            grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring=scoring[target],
        )
        search.fit(x["train"], y["train"])
        model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        for split in ["validation", "test"]:
            predicted = model.predict(x[split])
            if target == "target_congestion":
                metrics = multiclass_metrics(y[split], predicted, [0, 1, 2, 3])
                if split == "test":
                    print("Confusion matrix (test): rows = real class, columns = predicted class")
                    print(confusion_matrix(y[split], predicted, [0, 1, 2, 3]))
            else:
                metrics = binary_metrics(y[split], predicted, model.predict_proba(x[split])[:, 1])
            rows.append({"target": target, "model": "Decision Tree", "split": split, **metrics})
        save_fitted(model, "Decision Tree", target, feature_columns, train, rows)
    return pd.DataFrame(rows)

def tune_class_weights(probabilities: np.ndarray, y_true: np.ndarray, classes: list[int]) -> np.ndarray:
    weights = np.ones(len(classes))
    candidates = np.arange(0.5, 2.55, 0.1)
    for _ in range(3):
        for index in range(1, len(classes)):
            best_value, best_score = weights[index], -1.0
            for value in candidates:
                trial = weights.copy()
                trial[index] = value
                predicted = np.argmax(probabilities * trial, axis=1)
                score = multiclass_metrics(y_true, predicted, classes)["macro_F1"]
                if score > best_score:
                    best_value, best_score = value, score
            weights[index] = best_value
    return weights


def run_ensemble_classifiers(model_key: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = {
        "train": train[feature_columns].to_numpy(dtype=float),
        "validation": validation[feature_columns].to_numpy(dtype=float),
        "test": test[feature_columns].to_numpy(dtype=float),
    }
    scoring = {"target_congestion": "f1_macro", "target_accident": "roc_auc"}

    rows = []
    for target in CLASSIFICATION_TARGETS:
        y = {
            "train": train[target].to_numpy(dtype=int),
            "validation": validation[target].to_numpy(dtype=int),
            "test": test[target].to_numpy(dtype=int),
        }
        weights = compute_sample_weight("balanced", y["train"])
        models = [
            (
                "Random Forest",
                RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1),
                {"max_depth": [10, 16], "min_samples_leaf": [5, 20]},
                {},
            ),
            (
                "XGBoost",
                XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
                {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
                {"sample_weight": weights},
            ),
        ]
        models = [model for model in models if model[0] == ENSEMBLE_NAMES[model_key]]
        for name, estimator, grid, fit_params in models:
            print(f"\n===== {name} classifier: {target} =====")
            search = GridSearchCV(estimator, grid, cv=TimeSeriesSplit(n_splits=3), scoring=scoring[target])
            search.fit(x["train"], y["train"], **fit_params)
            model = search.best_estimator_
            print(f"Best parameters: {search.best_params_}")
            for split in ["validation", "test"]:
                predicted = model.predict(x[split])
                if target == "target_congestion":
                    metrics = multiclass_metrics(y[split], predicted, [0, 1, 2, 3])
                    if split == "test":
                        print("Confusion matrix (test): rows = real class, columns = predicted class")
                        print(confusion_matrix(y[split], predicted, [0, 1, 2, 3]))
                else:
                    metrics = binary_metrics(y[split], predicted, model.predict_proba(x[split])[:, 1])
                rows.append({"target": target, "model": name, "split": split, **metrics})
            save_fitted(model, name, target, feature_columns, train, rows)
            if target == "target_congestion":
                class_weights = tune_class_weights(model.predict_proba(x["validation"]), y["validation"], [0, 1, 2, 3])
                print(f"Class weights tuned on validation: {class_weights.round(2)}")
                for split in ["validation", "test"]:
                    predicted = np.argmax(model.predict_proba(x[split]) * class_weights, axis=1)
                    metrics = multiclass_metrics(y[split], predicted, [0, 1, 2, 3])
                    if split == "test":
                        print("Confusion matrix (test, tuned): rows = real class, columns = predicted class")
                        print(confusion_matrix(y[split], predicted, [0, 1, 2, 3]))
                    rows.append({"target": target, "model": f"{name} (tuned)", "split": split, **metrics})
                save_fitted(model, f"{name} (tuned)", target, feature_columns, train, rows, class_weights=class_weights)
    return pd.DataFrame(rows)


def add_prediction_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    calibration = predictions[predictions["split"] == "validation"]
    test = predictions[predictions["split"] == "test"]
    for label in ["volume", "travel_time"]:
        predicted, actual = f"predicted_{label}", f"actual_{label}"
        if predicted not in predictions:
            continue
        relative_error = (calibration[predicted] - calibration[actual]).abs() / calibration[predicted].clip(lower=1e-6)
        level = min(1.0, np.ceil((len(relative_error) + 1) * CONFIDENCE_LEVEL) / len(relative_error))
        width = relative_error.quantile(level)
        predictions[f"{predicted}_lower"] = predictions[predicted] * (1 - width)
        predictions[f"{predicted}_upper"] = predictions[predicted] * (1 + width)
        covered = ((test[actual] >= test[predicted] * (1 - width)) & (test[actual] <= test[predicted] * (1 + width))).mean()
        print(f"{label}: interval +/-{width:.1%}, covers {covered:.1%} of test actuals (target {CONFIDENCE_LEVEL:.0%})")

    if "congestion_confidence" in predictions:
        correct = test["predicted_congestion"] == test["actual_congestion"]
        sure = test["congestion_confidence"] >= 0.8
        print(f"congestion: {correct[sure].mean():.1%} correct when confidence >= 0.8 ({sure.mean():.0%} of windows), {correct[~sure].mean():.1%} otherwise")
    return predictions


def choose_winners() -> dict[str, str]:
    scoreboard = pd.concat([pd.read_csv(SCOREBOARD_FILE), pd.read_csv(CLASSIFICATION_FILE)], ignore_index=True)
    scoreboard = scoreboard[(scoreboard["split"] == "validation") & ~scoreboard["model"].str.contains(r"\(")]

    winners = {}
    for target, (metric, higher_is_better) in PRIMARY_METRICS.items():
        part = scoreboard[scoreboard["target"] == target].dropna(subset=[metric])
        saved = part[[model_path(name, target, "joblib").exists() for name in part["model"]]]
        if saved.empty:
            raise SystemExit(f"No saved model for {target}. Run the models first, for example: python3 src/ml_models.py classification forest")
        best = saved[metric].idxmax() if higher_is_better else saved[metric].idxmin()
        winners[target] = saved.loc[best, "model"]
        print(f"{target}: {winners[target]} ({metric} {saved.loc[best, metric]:.3f} on validation, chosen from {len(saved)} saved models)")
    return winners


def predict_windows(data: pd.DataFrame, winners: dict[str, str]) -> pd.DataFrame:
    output = data[["road_id", "timestamp"]].copy()
    output["forecast_for"] = output["timestamp"] + pd.Timedelta(minutes=30)
    for target, name in winners.items():
        bundle = load_model(name, target)
        x = data[bundle["feature_columns"]].to_numpy(dtype=float)
        if bundle["scaler"] is not None:
            x = apply_scaler(x, *bundle["scaler"])
        model = bundle["model"]
        label = target.removeprefix("target_")
        if target == "target_accident":
            output["accident_risk"] = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)
        elif target == "target_congestion":
            output["predicted_congestion"] = model.predict(x)
            if hasattr(model, "predict_proba"):
                output["congestion_confidence"] = model.predict_proba(x).max(axis=1)
        else:
            output[f"predicted_{label}"] = model.predict(x)
    names = {code: level for level, code in CONGESTION_ORDER.items()}
    output["predicted_congestion_level"] = output["predicted_congestion"].map(names)
    actual = data[TARGET_COLUMNS].rename(columns=lambda column: "actual_" + column.removeprefix("target_"))
    return pd.concat([output, actual], axis=1)


def parse_arguments(arguments: list[str]) -> tuple[str, str]:
    if arguments == ["predict"]:
        return "predict", ""
    if len(arguments) != 2 or arguments[0] not in MODEL_OPTIONS or arguments[1] not in MODEL_OPTIONS[arguments[0]]:
        for task, names in MODEL_OPTIONS.items():
            print(f"python3 src/ml_models.py {task} <{'|'.join(names)}>")
        print("python3 src/ml_models.py predict")
        sys.exit(1)
    return arguments[0], arguments[1]


def merge_scoreboard(file, new_rows: pd.DataFrame) -> pd.DataFrame:
    scoreboard = new_rows
    if file.exists():
        saved = pd.read_csv(file)
        replaced = saved.set_index(["target", "model"]).index.isin(new_rows.set_index(["target", "model"]).index)
        scoreboard = pd.concat([saved[~replaced], new_rows], ignore_index=True)
    file.parent.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(file, index=False)
    return scoreboard


def show_scoreboard(scoreboard: pd.DataFrame) -> None:
    for target in REGRESSION_TARGETS:
        for split in ["validation", "test"]:
            part = scoreboard[(scoreboard["target"] == target) & (scoreboard["split"] == split)]
            print(f"\n{target} - {split}")
            print(part[["model", "RMSE", "MAE", "MAPE", "R2"]].round(3).to_string(index=False))


def main():
    task, model_key = parse_arguments(sys.argv[1:])
    path = PROCESSED_DIRECTORY/f"features_{DATASET_VERSION}.csv"
    data = load_dataset(path)
    feature_columns = get_feature_columns(data)
    train, validation, test = time_split(data)

    if task == "predict":
        winners = choose_winners()
        predictions = pd.concat(
            [predict_windows(validation, winners).assign(split="validation"), predict_windows(test, winners).assign(split="test")],
            ignore_index=True,
        )
        predictions = add_prediction_intervals(predictions)
        predictions.to_csv(PREDICTIONS_FILE, index=False)
        print(f"Saved {len(predictions)} predictions to {PREDICTIONS_FILE.name}")
        return

    if task == "classification":
        classifiers = {
            "svm": lambda: run_svm(train, validation, test, feature_columns),
            "tree": lambda: run_decision_tree_classifier(train, validation, test, feature_columns),
            "forest": lambda: run_ensemble_classifiers("forest", train, validation, test, feature_columns),
            "xgboost": lambda: run_ensemble_classifiers("xgboost", train, validation, test, feature_columns),
        }
        scoreboard = merge_scoreboard(CLASSIFICATION_FILE, classifiers[model_key]())
        show_classification_scoreboard(scoreboard)
        return
    
    regressors = {
        "linear": lambda: run_linear_regression(train, validation, test, feature_columns),
        "tree": lambda: run_decision_tree(train, validation, test, feature_columns),
        "forest": lambda: run_random_forest(train, validation, test, feature_columns),
        "xgboost": lambda: run_xgboost(train, validation, test, feature_columns),
    }
    scoreboard = merge_scoreboard(SCOREBOARD_FILE, regressors[model_key]())
    show_scoreboard(scoreboard)

if __name__ == "__main__" :
    main()