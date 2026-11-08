import pandas as pd
from ingest import TIMEZONE,PROJECT_ROOT
import numpy as np
from features import TARGET_COLUMNS,DATASET_VERSION,PROCESSED_DIRECTORY
from sklearn.linear_model import LinearRegression
from evaluate import regression_metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor, export_text



SCOREBOARD_FILE = PROJECT_ROOT / "reports" / "regression_scoreboard.csv"

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
REGRESSION_TARGETS = ["target_volume", "target_travel_time"]
RANDOM_SEED = 42

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

        if target == "target_volume":
            print("\nTop rules of the tree:")
            print(export_text(model, feature_names=feature_columns, max_depth=3))
    return pd.DataFrame(rows)

def show_scoreboard(scoreboard: pd.DataFrame) -> None:
    for target in REGRESSION_TARGETS:
        for split in ["validation", "test"]:
            part = scoreboard[(scoreboard["target"] == target) & (scoreboard["split"] == split)]
            print(f"\n{target} - {split}")
            print(part[["model", "RMSE", "MAE", "MAPE", "R2"]].round(3).to_string(index=False))

def main():
    
    path = PROCESSED_DIRECTORY/f"features_{DATASET_VERSION}.csv"
    data = load_dataset(path)
    feature_columns = get_feature_columns(data)
    print(feature_columns)
    train, validation, test = time_split(data)
    linear_scoreboard = run_linear_regression(train,validation,test,feature_columns)
    tree_scoreboard = run_decision_tree(train,validation,test,feature_columns)
    scoreboard = pd.concat([linear_scoreboard, tree_scoreboard], ignore_index=True)
    show_scoreboard(scoreboard)

    SCOREBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(SCOREBOARD_FILE, index=False)

 



if __name__ == "__main__" :
    main()