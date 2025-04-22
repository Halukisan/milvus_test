import numpy as np

def reorder_clusters(clustered_results, query_vector, strategy="distance"):
    """
    对聚类结果进行重排序。
    """
    query_vector = np.array(query_vector)

    def compute_cluster_center(cluster):
        embeddings = np.array([result["embedding"] for result in cluster])
        return np.mean(embeddings, axis=0)

    if strategy == "distance":
        for cluster_label in clustered_results:
            clustered_results[cluster_label].sort(key=lambda x: x["distance"])
        sorted_clusters = clustered_results

    elif strategy == "cluster_size":
        sorted_clusters = dict(sorted(clustered_results.items(), key=lambda x: len(x[1]), reverse=True))

    elif strategy == "cluster_center":
        cluster_centers = {
            label: compute_cluster_center(cluster)
            for label, cluster in clustered_results.items()
        }
        sorted_labels = sorted(cluster_centers.keys(), key=lambda label: np.linalg.norm(cluster_centers[label] - query_vector))
        sorted_clusters = {label: clustered_results[label] for label in sorted_labels}

    else:
        raise ValueError(f"Unsupported reorder strategy: {strategy}")

    return sorted_clusters