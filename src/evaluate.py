import numpy as np
import pandas as pd


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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> np.ndarray:
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for row, actual in enumerate(classes):
        for column, predicted in enumerate(classes):
            matrix[row, column] = np.sum((y_true == actual) & (y_pred == predicted))
    return matrix


def per_class_scores(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_positive = np.diag(matrix).astype(float)
    predicted_total = matrix.sum(axis=0)
    actual_total = matrix.sum(axis=1)

    precision = np.divide(true_positive, predicted_total, out=np.zeros(len(matrix)), where=predicted_total > 0)
    recall = np.divide(true_positive, actual_total, out=np.zeros(len(matrix)), where=actual_total > 0)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros(len(matrix)), where=both > 0)
    return precision, recall, f1


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, classes)
    precision, recall, f1 = per_class_scores(matrix)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_F1": float(f1.mean()),
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    ranks = pd.Series(scores).rank().to_numpy()
    positive = y_true == 1
    positive_count = positive.sum()
    negative_count = len(y_true) - positive_count
    return float((ranks[positive].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, [0, 1])
    precision, recall, f1 = per_class_scores(matrix)
    return {
        "ROC_AUC": roc_auc(y_true, scores),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "F1": float(f1[1]),
    }