import pandas as pd
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import FieldSchema, Collection, connections, CollectionSchema, DataType
from modelscope import snapshot_download
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from pymilvus import Collection

def insert_milvus(file_path):
        
    load_dotenv()

    df = pd.read_csv(file_path) # 读取csv文件

    # 使用列表推导式，将dataFame的title和description列拼接成一个列表，每个字符串由标题和描述组成
    docs = [
        f"{title}\n{description}" for title, description in zip(df.title, df.description)
    ]

    # 使用modelscope下载模型
    model_path = snapshot_download('BAAI/bge-m3', revision='master')
    # 然后使用本地路径加载模型
    ef = BGEM3EmbeddingFunction(
        model_name=model_path,  # 使用本地模型路径
        device='cpu', # 指定设备为cpu
        use_fp16=False # 是否使用fp16精度
    )

    embeddings = ef(docs)["dense"] # ef为嵌入函数，docs为输入数据，dense键对应的值为嵌入向量

    connections.connect(uri="milvus.db") # 数据存入

    # 结构
    fields = [
        FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        ), 
        FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024
        ), 
        FieldSchema(
            name="text", dtype=DataType.VARCHAR, max_length=65535
        ), 
    ]

    schema = CollectionSchema(fields=fields, description="Embedding collection")

    collection = Collection(name="news_data", schema=schema)

    for doc, embedding in zip(docs, embeddings):
        collection.insert(
            {
                "text": doc, 
                "embedding": embedding
            }
        )

    index_params={
                "metric_type": "L2",
                "index_type": "GPU_BRUTE_FORCE"
            }

    collection.create_index(field_name="embedding", index_params=index_params)

    collection.flush()

    