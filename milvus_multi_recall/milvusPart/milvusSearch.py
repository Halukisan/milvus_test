
from pymilvus import Collection, db, connections
import numpy as np
from zhipuai import ZhipuAI
from pymilvus import MilvusClient
from dwspark.models import ChatModel
from typing import List
import gradio as gr # 通过as指定gradio库的别名为gr
import re
from urllib.parse import urlparse
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from modelscope import snapshot_download
import hdbscan

def extract_urls_with_positions(text):
    url_pattern = r'(https?://[^\s\)\]\}>]+)'
    return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]
def get_embedding(question):
    text  = [question]
    connections.connect(host="127.0.0.1", port=19530)
    db.using_database("news_data")
    coll_name = 'milvus_gpu'
    # 加载集合，不加载就不能插入数据
    collection = Collection(coll_name)
    collection.load()

    model_path = snapshot_download('BAAI/bge-m3', revision='master')
    # 然后使用本地路径加载模型
    ef = BGEM3EmbeddingFunction(
        model_name=model_path,  # 使用本地模型路径
        device='cpu', # 指定设备为cpu
        use_fp16=False # 是否使用fp16精度
    )

    embeddings = ef(text)["dense"] # ef为嵌入函数，docs为输入数据，dense键对应的值为嵌入向量

    # 查询迭代器
    # batch_size为每次查询的条数
    # expr为查询条件
    # output_fields为输出的字段
    iterator = collection.query_iterator(
        batch_size=10, expr="id > 0", output_fields=["id", "embedding"]
    )
    search_params = {
        "metric_type": "L2",
        "index_type": "GPU_BRUTE_FORCE",
        "limit": "6"
    }
    ids = []
    dist = {}

    embeddings = []

    while True:
        # 提取每条数据的id和embedding，id存入ids列表中
        batch = iterator.next()
        batch_ids = [data["id"] for data in batch]
        ids.extend(batch_ids)
        # 将每条数据的embedding存入embeddings列表中 
        query_vectors = [data["embedding"] for data in batch]
        embeddings.extend(query_vectors)
        # 现在获取到了ids列表和embeddings列表，我们可以使用hdbscan进行聚类操作了

        # 使用milvus的搜索
        results = collection.search(
            data=query_vectors,
            limit=6,
            anns_field="embedding",
            param=search_params,
            output_fields=["id"],
        )
        # 搜索结果存入dist字典中，键为batch_id，值为一个列表，列表中每个元素为一个元组，元组的第一个元素为id，第二个元素为距离（与该向量最相似的ID及其距离）
        for i, batch_id in enumerate(batch_ids):
            dist[batch_id] = []
            for result in results[i]:
                dist[batch_id].append((result.id, result.distance))

        if len(batch) == 0:
            break

    # 构建距离矩阵
    # ids2index是一个字典，键为id，值为该id在ids列表中的索引
    ids2index = {}

    for id in dist:
        ids2index[id] = len(ids2index)

    # dist_metric是二维距离矩阵
    dist_metric = np.full((len(ids), len(ids)), np.inf, dtype=np.float64)

    # 根据字典中的搜索结果填充距离矩阵，表示meita_id和batch_id之间的距离
    for id in dist:
        for result in dist[id]:
            dist_metric[ids2index[id]][ids2index[result[0]]] = result[1]

    # 使用HDBSCAN进行聚类，min_samples为每个点的最小邻居数，min_cluster_size为每个簇的最小点数，metric为距离度量方法，precomputed表示使用预计算的距离矩阵
    hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3, metric="precomputed")
    # 从milvus中批量的查询数据
    iterator = collection.query_iterator(
        batch_size=10, expr="id > 0", output_fields=["id", "text"]
    )

    ids = []
    texts = []
    # 查询到的数据存储到列表中
    while True:
        batch = iterator.next()
        if len(batch) == 0:
            break
        batch_ids = [data["id"] for data in batch]
        batch_texts = [data["text"] for data in batch]
        ids.extend(batch_ids)
        texts.extend(batch_texts)

    print(f"检索得到的答案为：{texts}")
    # 提取url和位置
    urls_with_positions = extract_urls_with_positions(texts)
    # 返回url和位置
    return texts, urls_with_positions[0][0] if urls_with_positions else ""


