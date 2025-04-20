import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from pymilvus import FieldSchema, Collection, connections, CollectionSchema, DataType

"""
聚类算法
"""

def Hdbscan(collection, search_params):
    """
    HDBSCAN聚类算法
    """
    iterator = collection.query_iterator(
        batch_size=10, expr="id > 0", output_fields=["id", "embedding"]
    )
    ids = []
    dist = {}
    embeddings = []

    while True:
        batch = iterator.next()
        if len(batch) == 0:
            break
        batch_ids = [data["id"] for data in batch]
        ids.extend(batch_ids)
        query_vectors = [data["embedding"] for data in batch]
        embeddings.extend(query_vectors)

        results = collection.search(
            data=query_vectors,
            limit=50,
            anns_field="embedding",
            param=search_params,
            output_fields=["id"],
        )
        for i, batch_id in enumerate(batch_ids):
            dist[batch_id] = []
            for result in results[i]:
                dist[batch_id].append((result.id, result.distance))

    # 构建距离矩阵
    ids2index = {id: idx for idx, id in enumerate(ids)}
    dist_metric = np.full((len(ids), len(ids)), 1e6, dtype=np.float64)  # 默认值为1e6
    for id in dist:
        for result in dist[id]:
            dist_metric[ids2index[id]][ids2index[result[0]]] = result[1]

    # 使用HDBSCAN进行聚类
    h = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3, metric="precomputed")
    hdb = h.fit(dist_metric)
    return hdb.labels_