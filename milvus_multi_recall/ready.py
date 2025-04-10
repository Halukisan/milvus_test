from ESPart.EsInsert import insert_elasticsearch
from milvusPart.milvusInsert import insert_milvus

def ready():
    file_path = 'milvus_HDBSCAN\\news_data_dedup.csv'
    # 1.插入数据到es
    insert_elasticsearch(file_path)
    # 2.插入数据到milvus
    insert_milvus(file_path)