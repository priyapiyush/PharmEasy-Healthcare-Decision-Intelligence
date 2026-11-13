import numpy as np

from shared import ROOT, load_records, standardize, write_csv, write_results


def kmeans(features, k):
    step = len(features) // k
    centroids = features[np.arange(k) * step].copy()
    labels = np.full(len(features), -1, dtype=int)
    while True:
        distances = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centroids[cluster] = np.mean(members, axis=0)
    distances = np.sum((features - centroids[labels]) ** 2, axis=1)
    return labels, centroids, float(np.sum(distances))


def hierarchical_rows(distance_matrix, linkage):
    clusters = [[index] for index in range(len(distance_matrix))]
    rows = []
    heights = []
    for merge_no in range(1, len(distance_matrix)):
        best_pair = None
        best_distance = None
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                pair_distances = [
                    distance_matrix[left, right]
                    for left in clusters[first]
                    for right in clusters[second]
                ]
                if linkage == "single":
                    value = min(pair_distances)
                elif linkage == "complete":
                    value = max(pair_distances)
                else:
                    value = float(np.mean(pair_distances))
                if best_distance is None or value < best_distance:
                    best_distance = value
                    best_pair = (first, second)
        first, second = best_pair
        merged = sorted(clusters[first] + clusters[second])
        cluster_a = sorted(clusters[first])
        cluster_b = sorted(clusters[second])
        if cluster_a[0] > cluster_b[0]:
            cluster_a, cluster_b = cluster_b, cluster_a
        format_cluster = lambda cluster: "+".join(f"E{member + 1}" for member in cluster)
        rows.append([linkage, merge_no, format_cluster(cluster_a), format_cluster(cluster_b), f"{best_distance:.4f}"])
        heights.append(float(best_distance))
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {first, second}]
        clusters.append(merged)
    gaps = np.diff(np.asarray(heights))
    cut_gap_index = int(np.argmax(gaps))
    merges_below_cut = cut_gap_index + 1
    cut_clusters = [[index] for index in range(len(distance_matrix))]
    for _ in range(merges_below_cut):
        best_pair = None
        best_distance = None
        for first in range(len(cut_clusters)):
            for second in range(first + 1, len(cut_clusters)):
                pair_distances = [
                    distance_matrix[left, right]
                    for left in cut_clusters[first]
                    for right in cut_clusters[second]
                ]
                if linkage == "single":
                    value = min(pair_distances)
                elif linkage == "complete":
                    value = max(pair_distances)
                else:
                    value = float(np.mean(pair_distances))
                if best_distance is None or value < best_distance:
                    best_distance = value
                    best_pair = (first, second)
        first, second = best_pair
        merged = sorted(cut_clusters[first] + cut_clusters[second])
        cut_clusters = [cluster for index, cluster in enumerate(cut_clusters) if index not in {first, second}]
        cut_clusters.append(merged)
    cut_clusters.sort(key=lambda cluster: cluster[0])
    cut_text = ";".join("+".join(f"E{member + 1}" for member in cluster) for cluster in cut_clusters)
    return rows, cut_text


def main():
    _, _, data = load_records()
    continuous = data[:, :5]
    outcome = data[:, 7].astype(int)
    value = data[:, 8]
    standardized, _, _ = standardize(continuous)

    wcss = {}
    for k in range(2, 7):
        _, _, wcss[k] = kmeans(standardized, k)
    elbow_k = next(
        k for k in (3, 4, 5)
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k])
    )

    labels, _, _ = kmeans(standardized, 3)
    segment_order = sorted(range(3), key=lambda cluster: float(np.mean(continuous[labels == cluster, 2])))
    segment_names = {cluster: chr(ord("A") + position) for position, cluster in enumerate(segment_order)}
    segment_rows = []
    segment_sizes = {}
    for cluster in segment_order:
        name = segment_names[cluster]
        members = continuous[labels == cluster]
        segment_sizes[name] = len(members)
        segment_rows.append([
            name, len(members),
            f"{np.mean(members[:, 0]):.4f}", f"{np.mean(members[:, 1]):.4f}",
            f"{np.mean(members[:, 2]):.4f}", f"{np.mean(members[:, 3]):.4f}",
            f"{np.mean(members[:, 4]):.4f}", f"{np.mean(outcome[labels == cluster]):.4f}",
            f"{np.mean(value[labels == cluster]):.4f}",
        ])
    write_csv(
        ROOT / "outputs" / "part3_segments.csv",
        ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
        segment_rows,
    )

    distance_matrix = np.asarray([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ])
    hierarchical_rows_all = []
    cut_values = {}
    for linkage in ("single", "complete", "average"):
        rows, cut_text = hierarchical_rows(distance_matrix, linkage)
        hierarchical_rows_all.extend(rows)
        cut_values[linkage] = cut_text
    write_csv(
        ROOT / "outputs" / "part3_hierarchical.csv",
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hierarchical_rows_all,
    )

    covariance = standardized.T @ standardized / (len(standardized) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for component in range(eigenvectors.shape[1]):
        largest = int(np.argmax(np.abs(eigenvectors[:, component])))
        if eigenvectors[largest, component] < 0:
            eigenvectors[:, component] *= -1
    projections = standardized @ eigenvectors[:, :2]
    eigenvalue_sum = float(np.sum(eigenvalues))
    explained = eigenvalues / eigenvalue_sum
    cumulative = np.cumsum(explained)
    components_for_80pct = int(np.argmax(cumulative >= 0.8) + 1)

    entries = [(f"wcss_k{k}", wcss[k]) for k in range(2, 7)]
    entries.extend([
        ("elbow_k", elbow_k), ("size_segment_A", segment_sizes["A"]),
        ("size_segment_B", segment_sizes["B"]), ("size_segment_C", segment_sizes["C"]),
        ("single_clusters_at_cut", cut_values["single"]),
        ("complete_clusters_at_cut", cut_values["complete"]),
        ("average_clusters_at_cut", cut_values["average"]),
        ("eigenvalue_sum", eigenvalue_sum),
    ])
    entries.extend((f"eigenvalue_{index + 1}", value_) for index, value_ in enumerate(eigenvalues))
    entries.extend((f"explained_ratio_{index + 1}", value_) for index, value_ in enumerate(explained))
    entries.extend((f"cumulative_ratio_{index + 1}", value_) for index, value_ in enumerate(cumulative))
    entries.append(("components_for_80pct", components_for_80pct))
    entries.extend((f"pc1_loading_X{index + 1}", eigenvectors[index, 0]) for index in range(5))
    entries.extend((f"pc2_loading_X{index + 1}", eigenvectors[index, 1]) for index in range(5))
    entries.extend([
        ("row1_pc1", projections[0, 0]), ("row1_pc2", projections[0, 1]),
        ("row2_pc1", projections[1, 0]), ("row2_pc2", projections[1, 1]),
        ("row3_pc1", projections[2, 0]), ("row3_pc2", projections[2, 1]),
    ])
    for name, cluster in segment_names.items():
        entries.append((f"mean_pc1_segment_{cluster}", float(np.mean(projections[labels == name, 0]))))
    write_results(ROOT / "outputs" / "part3_results.txt", entries)


if __name__ == "__main__":
    main()
