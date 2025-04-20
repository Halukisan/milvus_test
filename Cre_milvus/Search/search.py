import json
from pymilvus import Collection, db, connections
import numpy as np
from zhipuai import ZhipuAI
from pymilvus import MilvusClient
from dwspark.models import ChatModel
from typing import List
import gradio as gr # 通过as指定gradio库的别名为gr
from ColBuilder import Hdbscan
"""
搜索数据
"""

def search(VectorName, CollectionName, IP, Port, UserName, PassWord, question, topK, api_key):
    client = ZhipuAI(api_key=api_key)
    text = [question]
    connections.connect(VectorName, host=IP, port=Port, user=UserName, password=PassWord)
    collection = Collection(CollectionName)
    collection.load()

    response = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    if not collection.is_empty:
        resList = collection.search(
            anns_field="embedding",
            data=[response.data[0].embedding],
            limit=topK,
            param=search_params,
            output_fields=["id", "content", "embedding"]
        )
        return resList
    else:
        return []