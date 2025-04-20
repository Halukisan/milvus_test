import numpy as np


"""
聚类结果重排序
"""

def reorder_clusters(clustered_results, query_vector, strategy="distance"):
    """
    对聚类结果进行重排序。

    参数:
        clustered_results (dict): 聚类结果，格式为 {cluster_label: [result1, result2, ...]}。
        query_vector (list): 查询向量，用于计算相关性。
        strategy (str): 重排序策略，可选值为 "distance", "cluster_size", "cluster_center"。

    返回:
        dict: 重排序后的聚类结果。
    """
    query_vector = np.array(query_vector)

    # 计算聚类中心
    def compute_cluster_center(cluster):
        embeddings = np.array([result["embedding"] for result in cluster])
        return np.mean(embeddings, axis=0)

    # 按策略重排序
    if strategy == "distance":
        # 对每个聚类内部按距离排序
        for cluster_label in clustered_results:
            clustered_results[cluster_label].sort(key=lambda x: x["distance"])
        # 保持聚类顺序不变
        sorted_clusters = clustered_results

    elif strategy == "cluster_size":
        # 按聚类大小从大到小排序
        sorted_clusters = dict(sorted(clustered_results.items(), key=lambda x: len(x[1]), reverse=True))

    elif strategy == "cluster_center":
        # 按聚类中心与查询向量的距离排序
        cluster_centers = {
            label: compute_cluster_center(cluster)
            for label, cluster in clustered_results.items()
        }
        sorted_labels = sorted(cluster_centers.keys(), key=lambda label: np.linalg.norm(cluster_centers[label] - query_vector))
        sorted_clusters = {label: clustered_results[label] for label in sorted_labels}

    else:
        raise ValueError(f"Unsupported reorder strategy: {strategy}")

    return sorted_clusters