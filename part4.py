import math

import numpy as np

from shared import (
    ROOT,
    confusion_metrics,
    choose_knn_k,
    fit_logistic,
    knn_probabilities,
    load_records,
    logistic_probabilities,
    standardize,
    stratified_indices,
    write_csv,
    write_results,
)


def entropy(labels):
    labels = np.asarray(labels, dtype=int)
    total = len(labels)
    result = 0.0
    for count in (np.sum(labels == 0), np.sum(labels == 1)):
        if count:
            probability = count / total
            result -= probability * math.log2(probability)
    return result


def gini(labels):
    labels = np.asarray(labels, dtype=int)
    total = len(labels)
    return 1.0 - sum((np.sum(labels == value) / total) ** 2 for value in (0, 1))


def split_quality(features, target, condition):
    yes = np.asarray(condition, dtype=bool)
    no = ~yes
    parent_entropy = entropy(target)
    parent_gini = gini(target)
    yes_entropy = entropy(target[yes])
    no_entropy = entropy(target[no])
    yes_gini = gini(target[yes])
    no_gini = gini(target[no])
    weighted_entropy = len(target[yes]) / len(target) * yes_entropy + len(target[no]) / len(target) * no_entropy
    weighted_gini = len(target[yes]) / len(target) * yes_gini + len(target[no]) / len(target) * no_gini
    return {
        "n_yes": int(np.sum(yes)), "positives_yes": int(np.sum(target[yes])),
        "n_no": int(np.sum(no)), "positives_no": int(np.sum(target[no])),
        "info_gain": parent_entropy - weighted_entropy,
        "gini_gain": parent_gini - weighted_gini,
    }


def main():
    _, _, data = load_records()
    features = data[:, :7]
    target = data[:, 7].astype(int)
    train_indices, test_indices = stratified_indices(target)
    train_raw, test_raw = features[train_indices], features[test_indices]
    y_train, y_test = target[train_indices], target[test_indices]
    train_standard, train_mean, train_sd = standardize(train_raw)
    test_standard, _, _ = standardize(test_raw, train_mean, train_sd)

    logistic_weights, logistic_bias, _ = fit_logistic(train_standard, y_train, updates=1000)
    logistic_prob = logistic_probabilities(test_standard, logistic_weights, logistic_bias)
    best_k, _ = choose_knn_k(train_standard, y_train)
    knn_prob = knn_probabilities(train_standard, y_train, test_standard, best_k)

    split_conditions = {
        "S1": train_raw[:, 5] == 1,
        "S2": train_raw[:, 6] == 1,
        "S3": train_raw[:, 2] > 0.5,
        "S4": train_raw[:, 0] <= 24,
    }
    split_scores = {name: split_quality(train_raw, y_train, condition) for name, condition in split_conditions.items()}
    best_info = max(split_scores, key=lambda name: split_scores[name]["info_gain"])
    best_gini = max(split_scores, key=lambda name: split_scores[name]["gini_gain"])
    best_condition_train = split_conditions[best_info]
    yes_rate = float(np.mean(y_train[best_condition_train]))
    no_rate = float(np.mean(y_train[~best_condition_train]))
    test_condition = {
        "S1": test_raw[:, 5] == 1,
        "S2": test_raw[:, 6] == 1,
        "S3": test_raw[:, 2] > 0.5,
        "S4": test_raw[:, 0] <= 24,
    }[best_info]
    stump_prob = np.where(test_condition, yes_rate, no_rate)

    voting_predictions = {
        "logistic": logistic_prob >= 0.3,
        "knn": knn_prob >= 0.5,
        "stump": stump_prob >= 0.3,
        "hard_vote": ((logistic_prob >= 0.3).astype(int) + (knn_prob >= 0.5).astype(int) + (stump_prob >= 0.3).astype(int)) >= 2,
        "soft_vote": (np.mean(np.vstack((logistic_prob, knn_prob, stump_prob)), axis=0) >= 0.3),
    }
    voting_rows = []
    for model in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
        metrics = confusion_metrics(y_test, voting_predictions[model].astype(int))
        voting_rows.append([
            model, metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
            f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}",
        ])
    write_csv(ROOT / "outputs" / "part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], voting_rows)

    true_classes = np.asarray([1, 1, 1, -1, -1, -1, 1, -1])
    weak_one = np.asarray([1, 1, -1, -1, -1, -1, -1, -1])
    weak_two = np.asarray([-1, 1, 1, -1, 1, -1, 1, -1])
    weights = np.full(8, 1.0 / 8.0)
    adaboost_rows = []
    alphas = []
    errors = []
    for round_number, weak_rule in enumerate((weak_one, weak_two), start=1):
        weight_before = weights.copy()
        error = float(np.sum(weight_before[weak_rule != true_classes]))
        alpha = 0.5 * math.log((1.0 - error) / error)
        unnormalised = weight_before * np.exp(-alpha * true_classes * weak_rule)
        weights = unnormalised / np.sum(unnormalised)
        errors.append(error)
        alphas.append(alpha)
        for point in range(8):
            adaboost_rows.append([
                round_number, point + 1, f"{weight_before[point]:.4f}",
                f"{unnormalised[point]:.4f}", f"{weights[point]:.4f}",
            ])
    ensemble_score = alphas[0] * weak_one + alphas[1] * weak_two
    ensemble_accuracy = float(np.mean(np.sign(ensemble_score) == true_classes))
    write_csv(ROOT / "outputs" / "part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], adaboost_rows)

    mlp_layers = {
        "mlp_hidden1": (784 * 16) + 16,
        "mlp_hidden2": (16 * 16) + 16,
        "mlp_output": (16 * 10) + 10,
    }
    cnn_layers = {
        "cnn_conv1": (5 * 5 * 1 * 6) + 6,
        "cnn_conv2": (5 * 5 * 6 * 16) + 16,
        "cnn_dense1": (5 * 5 * 16 * 32) + 32,
        "cnn_dense2": (32 * 10) + 10,
    }
    architecture_rows = [
        ["mlp_hidden1", mlp_layers["mlp_hidden1"]],
        ["mlp_hidden2", mlp_layers["mlp_hidden2"]],
        ["mlp_output", mlp_layers["mlp_output"]],
        ["cnn_conv1", cnn_layers["cnn_conv1"]],
        ["cnn_conv2", cnn_layers["cnn_conv2"]],
        ["cnn_dense1", cnn_layers["cnn_dense1"]],
        ["cnn_dense2", cnn_layers["cnn_dense2"]],
    ]
    write_csv(ROOT / "outputs" / "part4_architecture.csv", ["layer", "parameters"], architecture_rows)
    fc_layer_params = (128 * 128 * 3 * 1024) + 1024
    conv_layer_params = (3 * 3 * 3 * 64) + 64
    rnn_total = (64 * 128) + (128 * 128) + (128 * 16) + 128 + 16

    entries = [
        ("root_entropy", entropy(y_train)), ("root_gini", gini(y_train)),
    ]
    split_rows = []
    for name in ("S1", "S2", "S3", "S4"):
        score = split_scores[name]
        split_rows.append([
            name, score["n_yes"], score["positives_yes"], score["n_no"], score["positives_no"],
            f"{score['info_gain']:.4f}", f"{score['gini_gain']:.4f}",
        ])
    write_csv(ROOT / "outputs" / "part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], split_rows)
    entries.extend([
        ("best_split_by_info_gain", best_info), ("best_split_by_gini_gain", best_gini),
        ("stump_rate_yes", yes_rate), ("stump_rate_no", no_rate),
        ("round1_error", errors[0]), ("round1_alpha", alphas[0]),
        ("round2_error", errors[1]), ("round2_alpha", alphas[1]),
        ("ensemble_accuracy", ensemble_accuracy),
        ("oob_fraction", (1.0 - 1.0 / 560) ** 560),
        ("mlp_total", sum(mlp_layers.values())), ("cnn_total", sum(cnn_layers.values())),
        ("fc_layer_params", fc_layer_params), ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ])
    write_results(ROOT / "outputs" / "part4_results.txt", entries)


if __name__ == "__main__":
    main()
