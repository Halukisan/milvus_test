import pymilvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from zhipuai import ZhipuAI
import os
import logging
import re
import time
from urllib.parse import urlparse

"""
索引构建，选择CPU还是GPU来构建索引
"""
def indexParam(Choice,IndexName):
    if Choice == "gpu":
        index_params={
        "metric_type": "L2",
        "index_type": IndexName,
        "params": {
            'intermediate_graph_degree': 64,
            'graph_degree': 32
        },
        "build_algo":"IVF_PQ",
        "cache_data_set_on_device":"true"
    }
    elif Choice == "cpu":
        index_params={
            "metric_type": "IP",
            "index_type": IndexName,
            "params": {"nlist": 1024}
        }
    
       
        """
        search_params的参数：
        intermediate_graph_degree(int)：通过在剪枝之前确定图的度数来影响召回率和构建时间。推荐值为32或64。
        graph_degree（int）：通过设置剪枝后图形的度数来影响搜索性能和召回率。通常，它是中间图度的一半。这两个度数之间的差值越大，构建时间就越长。它的值必须小于intermediate_graph_degree 的值。
        build_algo（字符串）：选择剪枝前的图形生成算法。可能的选项：
        IVF_PQ：提供更高的质量，但构建时间较慢
        NN_DESCENT：提供更快的生成速度，但可能会降低召回率。
        cache_dataset_on_device（字符串，"true"|"false"）：决定是否在 GPU 内存中缓存原始数据集。将其设置为"true "可通过细化搜索结果提高召回率，而将其设置为"false "则可节省 GPU 内存。
        """
    return index_params