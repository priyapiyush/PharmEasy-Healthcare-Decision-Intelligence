import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_records():
    with (ROOT / "data" / "records.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    data = rows[1:]
    ids = [row[0] for row in data]
    values = np.asarray([[float(item) for item in row[1:]] for row in data], dtype=float)
    return header, ids, values


def sample_sd(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum((values - np.mean(values)) ** 2) / (len(values) - 1)))


def standardize(values, mean=None, sd=None):
    values = np.asarray(values, dtype=float)
    if mean is None:
        mean = np.mean(values, axis=0)
    if sd is None:
        sd = np.sqrt(np.sum((values - mean) ** 2, axis=0) / (len(values) - 1))
    return (values - mean) / sd, mean, sd


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    clipped = np.clip(z, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_cross_entropy(y, probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities)))


def fit_logistic(features, target, learning_rate=0.1, updates=1000):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    losses = {0: binary_cross_entropy(target, sigmoid(features @ weights + bias))}
    for epoch in range(1, updates + 1):
        probabilities = sigmoid(features @ weights + bias)
        error = probabilities - target
        weights -= learning_rate * (features.T @ error) / len(target)
        bias -= learning_rate * float(np.sum(error)) / len(target)
        if epoch in {1, 10, 100, 200, 1000, updates}:
            losses[epoch] = binary_cross_entropy(target, sigmoid(features @ weights + bias))
    return weights, bias, losses


def logistic_probabilities(features, weights, bias):
    return sigmoid(np.asarray(features) @ weights + bias)


def stratified_indices(target):
    target = np.asarray(target, dtype=int)
    zero_indices = np.flatnonzero(target == 0)
    one_indices = np.flatnonzero(target == 1)
    zero_cut = (7 * len(zero_indices)) // 10
    one_cut = (7 * len(one_indices)) // 10
    train_set = set(zero_indices[:zero_cut].tolist() + one_indices[:one_cut].tolist())
    train_indices = np.asarray([index for index in range(len(target)) if index in train_set], dtype=int)
    test_indices = np.asarray([index for index in range(len(target)) if index not in train_set], dtype=int)
    return train_indices, test_indices


def confusion_metrics(target, predictions):
    target = np.asarray(target, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(np.sum((predictions == 1) & (target == 1)))
    fp = int(np.sum((predictions == 1) & (target == 0)))
    tn = int(np.sum((predictions == 0) & (target == 0)))
    fn = int(np.sum((predictions == 0) & (target == 1)))
    accuracy = (tp + tn) / len(target)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "fpr": fpr, "tpr": recall, "f1": f1,
    }


def knn_probabilities(reference_features, reference_target, query_features, k):
    reference_features = np.asarray(reference_features, dtype=float)
    reference_target = np.asarray(reference_target, dtype=int)
    query_features = np.asarray(query_features, dtype=float)
    probabilities = []
    for query in query_features:
        distances = np.sum((reference_features - query) ** 2, axis=1)
        order = np.argsort(distances, kind="stable")
        neighbours = order[:k]
        probabilities.append(float(np.mean(reference_target[neighbours])))
    return np.asarray(probabilities, dtype=float)


def choose_knn_k(features, target):
    best_k = None
    best_score = -1.0
    scores = []
    for k in range(1, 22, 2):
        fold_scores = []
        for fold in range(5):
            validation_mask = np.arange(len(target)) % 5 == fold
            probabilities = knn_probabilities(
                features[~validation_mask], target[~validation_mask], features[validation_mask], k,
            )
            metrics = confusion_metrics(target[validation_mask], (probabilities >= 0.5).astype(int))
            fold_scores.append(metrics["f1"])
        mean_f1 = float(np.mean(fold_scores))
        scores.append((k, mean_f1))
        if mean_f1 > best_score:
            best_score = mean_f1
            best_k = k
    return best_k, scores


def write_results(path, entries):
    with Path(path).open("w", newline="") as handle:
        for key, value in entries:
            if isinstance(value, (bool, np.bool_)):
                rendered = "yes" if value else "no"
            elif isinstance(value, (int, np.integer)):
                rendered = str(int(value))
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):.4f}"
            else:
                rendered = str(value)
            handle.write(f"{key} = {rendered}\n")


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def csv_float(value):
    return f"{float(value):.4f}"
