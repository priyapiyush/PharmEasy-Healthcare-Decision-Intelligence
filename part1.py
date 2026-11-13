import math

import numpy as np

from shared import ROOT, load_records, sample_sd, write_results


def simple_regression(x, y):
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    covariance = float(np.sum((x - x_mean) * (y - y_mean)) / (len(x) - 1))
    variance = float(np.sum((x - x_mean) ** 2) / (len(x) - 1))
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    return intercept, slope, fitted


def residual_sd(values):
    return sample_sd(values)


def main():
    _, _, data = load_records()
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    x4 = data[:, 3]
    x5 = data[:, 4]
    x6 = data[:, 5]
    x7 = data[:, 6]
    value = data[:, 8]
    n = len(value)
    mean_value = float(np.mean(value))
    sd_value = sample_sd(value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    reg_intercept, reg_slope, fitted = simple_regression(x4, value)
    residuals = value - fitted
    tss = float(np.sum((value - mean_value) ** 2))
    ess = float(np.sum((fitted - mean_value) ** 2))
    rss = float(np.sum(residuals ** 2))
    tss_equals_ess_plus_rss = abs(tss - ess - rss) < 1e-6 * tss
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(residuals)))

    x4_mean = float(np.mean(x4))
    x4_sd = sample_sd(x4)
    x4_z = (x4 - x4_mean) / x4_sd
    m = 0.0
    c = 0.0
    gd_costs = {0: float(np.mean(value ** 2))}
    for epoch in range(1, 501):
        prediction = m * x4_z + c
        error = value - prediction
        gradient_m = -2.0 * float(np.sum(x4_z * error)) / n
        gradient_c = -2.0 * float(np.sum(error)) / n
        m -= 0.1 * gradient_m
        c -= 0.1 * gradient_c
        if epoch in {1, 10, 100, 500}:
            gd_costs[epoch] = float(np.mean((value - (m * x4_z + c)) ** 2))
    gd_slope = m / x4_sd
    gd_intercept = c - m * x4_mean / x4_sd
    gd_slope_gap = abs(gd_slope - reg_slope)

    order = np.argsort(fitted, kind="stable")
    raw_lower = residuals[order[: n // 2]]
    raw_upper = residuals[order[n // 2 :]]
    resid_sd_lower_raw = residual_sd(raw_lower)
    resid_sd_upper_raw = residual_sd(raw_upper)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    log_x4 = np.log(x4)
    log_value = np.log(value)
    loglog_intercept, loglog_slope, loglog_fitted = simple_regression(log_x4, log_value)
    loglog_residuals = log_value - loglog_fitted
    loglog_tss = float(np.sum((log_value - np.mean(log_value)) ** 2))
    loglog_rss = float(np.sum(loglog_residuals ** 2))
    loglog_r2 = 1.0 - loglog_rss / loglog_tss
    loglog_rmse = math.sqrt(loglog_rss / n)
    log_order = np.argsort(loglog_fitted, kind="stable")
    log_lower = loglog_residuals[log_order[: n // 2]]
    log_upper = loglog_residuals[log_order[n // 2 :]]
    resid_sd_lower_log = residual_sd(log_lower)
    resid_sd_upper_log = residual_sd(log_upper)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    design = np.column_stack((
        np.ones(n), log_x4, x1, x2, x3, x5, x6, x7,
    ))
    beta = np.linalg.solve(design.T @ design, design.T @ log_value)
    mr_fitted = design @ beta
    mr_rss = float(np.sum((log_value - mr_fitted) ** 2))
    mr_r2 = 1.0 - mr_rss / loglog_tss
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    entries = [
        ("n", n), ("mean_value", mean_value), ("sd_value", sd_value),
        ("se_value", se_value), ("ci95_low", ci95_low), ("ci95_high", ci95_high),
        ("z_stat", z_stat), ("reject_h0", reject_h0), ("n_required", n_required),
        ("reg_slope", reg_slope), ("reg_intercept", reg_intercept),
        ("tss", tss), ("ess", ess), ("rss", rss),
        ("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss), ("r2", r2),
        ("rmse", rmse), ("mae", mae), ("gd_J0", gd_costs[0]),
        ("gd_J1", gd_costs[1]), ("gd_J10", gd_costs[10]),
        ("gd_J100", gd_costs[100]), ("gd_J500", gd_costs[500]),
        ("gd_slope", gd_slope), ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap), ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw), ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", loglog_intercept), ("loglog_slope", loglog_slope),
        ("loglog_r2", loglog_r2), ("loglog_rmse", loglog_rmse),
        ("resid_sd_lower_log", resid_sd_lower_log), ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
    ]
    entries.extend((f"mr_beta_{index}", coefficient) for index, coefficient in enumerate(beta))
    entries.extend([("mr_r2", mr_r2), ("mr_adj_r2", mr_adj_r2), ("loglog_adj_r2", loglog_adj_r2)])
    write_results(ROOT / "outputs" / "part1_results.txt", entries)


if __name__ == "__main__":
    main()
