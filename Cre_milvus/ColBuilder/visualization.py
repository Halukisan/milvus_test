import pandas as pd
from umap import UMAP

def get_cluster_visualization_data(embeddings, labels, texts):
    """
    embeddings: np.ndarray, shape=(n_samples, n_features)
    labels: array-like, shape=(n_samples,)
    texts: list of str, shape=(n_samples,)
    返回适合前端可视化的DataFrame
    """
    # UMAP降维
    umap = UMAP(n_components=2, random_state=42, n_neighbors=80, min_dist=0.1)
    umap_result = umap.fit_transform(embeddings)
    df = pd.DataFrame(umap_result, columns=["x", "y"])
    df["cluster"] = labels.astype(str)
    df["text"] = texts
    # 过滤噪声点
    df = df[df["cluster"] != "-1"].sort_values(by="cluster")
    return df