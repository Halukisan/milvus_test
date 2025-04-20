from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from zhipuai import ZhipuAI
import hdbscan
import os
import logging
import re
def milvus_connect(IP, Port, UserName, Password, VectorName, CollectionName, IndexParam, ReplicaNum, dataList, url_split):
    try:
        # 连接Milvus服务器
        connections.connect(VectorName, host=IP, port=Port, user=UserName, password=Password)

        # 检查并创建collection
        collection_name = CollectionName
        if url_split:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="cluster_label", dtype=DataType.INT64)
            ]
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="cluster_label", dtype=DataType.INT64)
            ]
        schema = CollectionSchema(fields, collection_name)
        collection = Collection(name=collection_name, schema=schema)

        # 聚类
        embeddings = [data["embedding"] for data in dataList]
        clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)

        # 插入数据
        for i, data in enumerate(dataList):
            data["cluster_label"] = labels[i]
        collection.insert(dataList)

        # 创建索引
        collection.create_index(field_name="embedding", index_params=IndexParam)

        # 启用内存副本
        collection.load(replica_number=ReplicaNum)
        return True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False
