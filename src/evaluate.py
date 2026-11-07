import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    squared_error_sum = np.sum(errors ** 2)
    total_variation = np.sum((y_true - y_true.mean()) ** 2)

    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
        "MAPE": float(np.mean(np.abs(errors / y_true)) * 100),
        "R2": float(1 - squared_error_sum / total_variation),
    }