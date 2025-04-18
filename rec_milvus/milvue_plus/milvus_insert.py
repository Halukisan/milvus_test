from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from zhipuai import ZhipuAI
import os
import logging
import re
# 设置日志级别
logging.basicConfig(level=logging.INFO)
def insert(schema_name,doc_entry):
        
    try:
        # 连接Milvus服务器
        connections.connect("default", host="localhost", port="19530")

        # 检查并创建collection
        collection_name = 'milvus_plus'
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="question_id", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        schema = CollectionSchema(fields, schema_name)
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
        collection.load()


        # 将数据插入Milvus
        mr = collection.insert(doc_entry)
        logging.info(f"Insert result: {mr}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        # 关闭Milvus连接
        connections.disconnect("default")
