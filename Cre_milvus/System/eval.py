import numpy as np
from pymilvus import Collection

def evaluate_data_quality(embedding):
    """
    简单的向量质量评估函数。
    """
    return np.linalg.norm(embedding) > 0.1  # 示例：向量模长大于 0.1 视为高质量

def insert_with_quality_check(collection, dataList):
    high_quality = []
    low_quality = []

    for data in dataList:
        if evaluate_data_quality(data["embedding"]):
            high_quality.append(data)
        else:
            low_quality.append(data)

    collection.insert(high_quality)
    low_quality_collection = Collection("low_quality_data")
    low_quality_collection.insert(low_quality)