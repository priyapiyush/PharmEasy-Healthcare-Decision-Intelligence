import numpy as np

from shared import (
    ROOT,
    binary_cross_entropy,
    choose_knn_k,
    confusion_metrics,
    fit_logistic,
    knn_probabilities,
    load_records,
    logistic_probabilities,
    standardize,
    stratified_indices,
    write_csv,
    write_results,
)


def main():
    _, _, data = load_records()
    features = data[:, :7]
    target = data[:, 7].astype(int)
    train_indices, test_indices = stratified_indices(target)
    x_train_raw, x_test_raw = features[train_indices], features[test_indices]
    y_train, y_test = target[train_indices], target[test_indices]
    x_train, train_mean, train_sd = standardize(x_train_raw)
    x_test, _, _ = standardize(x_test_raw, train_mean, train_sd)

    weights, bias, losses = fit_logistic(x_train, y_train, updates=1000)
    train_probabilities = logistic_probabilities(x_train, weights, bias)
    test_probabilities = logistic_probabilities(x_test, weights, bias)

    eta_losses = {}
    for eta in (0.01, 0.1, 1.0):
        _, _, eta_history = fit_logistic(x_train, y_train, learning_rate=eta, updates=200)
        eta_losses[eta] = eta_history[200]

    entries = [
        ("train_rows", len(train_indices)), ("train_positives", int(np.sum(y_train))),
        ("test_rows", len(test_indices)), ("test_positives", int(np.sum(y_test))),
        ("loss_epoch_0", losses[0]), ("loss_epoch_1", losses[1]),
        ("loss_epoch_10", losses[10]), ("loss_epoch_100", losses[100]),
        ("loss_epoch_1000", losses[1000]),
    ]
    entries.extend((f"weight_X{index + 1}", weights[index]) for index in range(7))
    entries.extend([
        ("bias", bias), ("test_bce", binary_cross_entropy(y_test, test_probabilities)),
        ("loss_epoch_200_eta_0.01", eta_losses[0.01]),
        ("loss_epoch_200_eta_0.1", eta_losses[0.1]),
        ("loss_epoch_200_eta_1.0", eta_losses[1.0]),
    ])
    entries.extend((f"odds_ratio_X{index + 1}", float(np.exp(weights[index]))) for index in range(7))

    threshold_metrics = {}
    for threshold in (0.5, 0.3):
        metrics = confusion_metrics(y_test, (test_probabilities >= threshold).astype(int))
        threshold_metrics[threshold] = metrics
        prefix = f"tau_{threshold:.1f}_"
        entries.extend((prefix + name, metrics[name]) for name in (
            "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity",
        ))
    blind_accuracy = float(np.mean(y_test == 0))
    entries.append(("blind_accuracy", blind_accuracy))

    threshold_rows = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        metrics = confusion_metrics(y_test, (test_probabilities >= threshold).astype(int))
        threshold_rows.append([
            f"{threshold:.1f}", metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
            f"{metrics['tpr']:.4f}", f"{metrics['fpr']:.4f}",
        ])
    write_csv(ROOT / "outputs" / "part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], threshold_rows)

    best_k, cv_scores = choose_knn_k(x_train, y_train)
    cv_rows = [[k, f"{mean_f1:.4f}"] for k, mean_f1 in cv_scores]
    write_csv(ROOT / "outputs" / "part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)

    knn_test_probabilities = knn_probabilities(x_train, y_train, x_test, best_k)
    knn_metrics = confusion_metrics(y_test, (knn_test_probabilities >= 0.5).astype(int))
    entries.append(("knn_best_k", best_k))
    entries.extend((f"knn_{name}", knn_metrics[name]) for name in (
        "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1",
    ))

    chosen_tau = None
    chosen_cost = None
    for numerator in range(1, 20):
        threshold = numerator / 20.0
        metrics = confusion_metrics(y_train, (train_probabilities >= threshold).astype(int))
        cost = 50 * metrics["fp"] + 400 * metrics["fn"]
        if chosen_cost is None or cost < chosen_cost:
            chosen_tau = threshold
            chosen_cost = cost
    test_at_tau = confusion_metrics(y_test, (test_probabilities >= chosen_tau).astype(int))
    test_at_half = threshold_metrics[0.5]
    entries.extend([
        ("cost_tau", chosen_tau), ("cost_train_at_tau", chosen_cost),
        ("cost_test_at_tau", 50 * test_at_tau["fp"] + 400 * test_at_tau["fn"]),
        ("cost_test_at_0.5", 50 * test_at_half["fp"] + 400 * test_at_half["fn"]),
        ("cost_test_flag_nobody", 400 * int(np.sum(y_test))),
    ])
    write_results(ROOT / "outputs" / "part2_results.txt", entries)


if __name__ == "__main__":
    main()
