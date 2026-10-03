import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)

    return float(np.sqrt(np.mean(err**2)))


def precision_recall_f1(true_edges, reconstructed_edges, observed_edges):

    hidden = set(true_edges) - set(observed_edges)

    reconstructed = set(reconstructed_edges)

    tp = len(reconstructed & hidden)

    fp = len(reconstructed - set(true_edges))

    fn = len(hidden - reconstructed)

    precision = tp / (tp + fp) if tp + fp else 0.0

    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
