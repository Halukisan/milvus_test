import json
from pymilvus import Collection, db, connections
import numpy as np
from zhipuai import ZhipuAI
from pymilvus import MilvusClient
from dwspark.models import ChatModel
from typing import List
import gradio as gr # 通过as指定gradio库的别名为gr


"""
搜索数据
"""

def search(VectorName,CollectionName,IP,Port,UserName,PassWord,question,topK,api_key):
    client = ZhipuAI(api_key=api_key)
    text  = [question]
    connections.connect(VectorName,host=IP, port=Port, user=UserName, password=PassWord)
    collection = Collection(CollectionName)
    collection.load()

    response = client.embeddings.create(
        model="embedding-2", 
        input=text 
    )
    collection = Collection(CollectionName)
    search_params = {
        # 这里后期可以定制
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }    
    if not collection.is_empty:
        print("开始查询")
        res = collection.search(
            anns_field="embeding",
            data=[response.data[0].embedding],
            limit=topK,
            param=search_params, # 搜索参数
            output_fields=["content"] # 返回的输出字段
        )
        return res
    return None