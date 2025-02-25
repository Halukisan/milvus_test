# 用于初始化计算中心向量
from create import create
from create import custom_text_splitter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from zhipuai import ZhipuAI
import os
import logging
import re
java_doc = []
vectors = []
api_key = ""
if not api_key:
    raise ValueError("Environment variable ZHIPUAI_API_KEY is not set.")
java_doc = []
# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)
def dataMake(Name):
        # 指定要读取的 Markdown 文件路径(示例为获取java类所有数据)
    markdown_file_path = f"../data_base/milvus_plus/${Name}"
    # 列出目录中的所有文件
    files_in_directory = os.listdir(markdown_file_path)

    # 过滤出所有的Markdown文件
    markdown_files = [file for file in files_in_directory if file.endswith('.md')]

    # 读取每个Markdown文件的内容
    for markdown_file in markdown_files:
        file_path = os.path.join(markdown_file_path, markdown_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            segments = custom_text_splitter(content)
            
            for i, segment in enumerate(segments, 1):
                non_url_text = segment

                non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()
                # 调用嵌入创建 API
                response = client.embeddings.create(
                    model="embedding-2",  # 替换为您想要使用的模型名称
                    input=non_url_text  # 传递单个文本
                )

                # 获取生成的 embedding 值
                embedding_value = response.data[0].embedding
                vectors.append(embedding_value)
def k_mean(Name):
    dataMake(Name)
    # 应用K-means进行聚类
    num_clusters = 10  # 根据实际情况调整聚类数量
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(vectors)

    # 获取每个聚类的中心点
    cluster_centers = kmeans.cluster_centers_

    # 使用所有聚类中心的平均值作为标准向量
    standard_vector = np.mean(cluster_centers, axis=0)
    # 重置vectors
    vectors= []
    print(f"{Name}聚类向量的中心平均向量（标准向量）:")
    print(standard_vector)
    
def init():
    java_keyVectors = k_mean("java")
    mysql_keyVectors = k_mean("mysql")
    redis_keyVectors = k_mean("redis")
    web_keyVectors = k_mean("web")

    create(java_keyVectors,mysql_keyVectors,redis_keyVectors,web_keyVectors)