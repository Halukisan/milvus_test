import pandas as pd
from elasticsearch import Elasticsearch, helpers

def insert_elasticsearch(file_path):
    df = pd.read_csv(file_path) # 读取csv文件
    # 准备数据
    docs = [
        {
            "_index": "dfs_query_then_fetch",  # 替换为你的索引名称
            "_source": {
                "title": title,
                "description": description,
                "content": f"{title}\n{description}"
            }
        }
        for title, description in zip(df.title, df.description)
    ]

    # 连接到Elasticsearch
    es_client = Elasticsearch(["http://localhost:9200"])

    # 检查索引是否存在，如果不存在则创建索引
    index_name = "dfs_query_then_fetch"  # 替换为你的索引名称
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "content": {"type": "text"}
                }
            }
        })

    # 插入文档
    helpers.bulk(es_client, docs)

    print(f"Successfully indexed {len(docs)} documents into {index_name}")



