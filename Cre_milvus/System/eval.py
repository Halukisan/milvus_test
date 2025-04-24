import numpy as np
from pymilvus import Collection

def evaluate_data_quality(txt):
    """
    简单的向量质量评估函数。牛牛你需要做的是判断这个txt（一条文本数据）的质量如何，如果质量高就返回True，
    否则返回False。
    """
    return None

def insert_with_quality_check(collection, dataList):
    high_quality = []
    low_quality = []

    for data in dataList:
        if evaluate_data_quality(data["content"]):
            high_quality.append(data)
        else:
            low_quality.append(data)

    status1 = collection.insert(high_quality)
    # 低质量数据插入到另一个集合中
    low_quality_collection = Collection("low_quality_data")
    status2 = low_quality_collection.insert(low_quality)
    return status1, status2


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def score_redis_recall(query_embedding, recall_results, score_key="embedding"):
    """
    对redis召回的每条数据，计算与query_embedding的相似度分数。
    :param query_embedding: 查询的embedding向量
    :param recall_results: redis召回的结果（list，每个元素为dict，需包含embedding字段）
    :param score_key: 召回结果中embedding字段名
    :return: 带有score字段的结果列表
    """
    scored = []
    for item in recall_results:
        item_embedding = item.get(score_key)
        if item_embedding is not None:
            score = cosine_similarity(query_embedding, item_embedding)
            item = dict(item) 
            item["score"] = score
            scored.append(item)
    return scored    