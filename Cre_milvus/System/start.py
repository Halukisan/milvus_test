from System.init import init_milvus, init_es, init_redis
from dataBuilder.data import data_process
from milvusBuilder.milvus import milvus_connect
from IndexParamBuilder.indexparam import indexParams
from reorder.reo_clu import reorder_clusters
from Search.search import search
from System.monitor import log_event
from System.Retry import retry_function
from System.eval import insert_with_quality_check
from Search.es_utils import create_index, bulk_insert_to_es
from Search.embedding import EmbeddingGenerator
import hdbscan
from sklearn.cluster import KMeans
import numpy as np
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

def Cre_VectorDataBaseStart_from_config(config):
    milvus_cfg = config["milvus"]
    data_cfg = config["data"]
    embedding_cfg = config["embedding"]

    return Cre_VectorDataBaseStart(
        C_G_Choic=milvus_cfg.get("index_device", "cpu"),
        IP=milvus_cfg["host"],
        Port=milvus_cfg["port"],
        UserName=milvus_cfg["user"],
        PassWord=milvus_cfg["password"],
        VectorName=milvus_cfg["vector_name"],
        CollectionName=milvus_cfg["collection_name"],
        IndexName=milvus_cfg["index_name"],
        ReplicaNum=milvus_cfg["replica_num"],
        Data_Location=data_cfg["location"],
        url_split=data_cfg["url_split"],
        embedding_model=embedding_cfg["model"],
        api_key=embedding_cfg["api_key"]
    )

def Cre_VectorDataBaseStart(
    C_G_Choic, IP, Port, UserName, PassWord, VectorName, CollectionName,
    IndexName, ReplicaNum, Data_Location, url_split, embedding_model, api_key
):
    """
    构建向量数据库并插入数据，参数全部由配置文件自动读取。
    """
    # 初始化连接
    init_milvus(VectorName, IP, Port, UserName, PassWord)
    init_es("http://localhost:9200")
    init_redis(host="localhost", port=6379)
    log_event("开始数据处理")
    # 数据处理加重试
    dataList = retry_function(
        lambda: data_process(
            data_location=Data_Location,
            model_name=embedding_model,
            api_key=api_key,
            url_split=url_split
        )
    )
    log_event(f"数据处理完成，数据量：{len(dataList)}")
    # === 新增：写入ES ===

    try:
        from elasticsearch import Elasticsearch
        es_client = Elasticsearch("http://localhost:9200")
        index_name = "doc_index"
        create_index(es_client, index_name)
        embedder = EmbeddingGenerator()
        for doc in dataList:
            # 若已存在embedding则跳过
            if "embedding" not in doc:
                doc["embedding"] = embedder.get_embedding(doc["content"])
        bulk_insert_to_es(es_client, index_name, dataList)
        log_event("ES写入完成")
    except Exception as e:
        log_event(f"ES写入失败: {e}")

    # 构建索引参数
    indexParam = indexParams(C_G_Choic, IndexName)

    # 连接Milvus并插入数据
    log_event("开始连接Milvus并插入数据")
    def milvus_insert():
        status, collection = milvus_connect(
            IP, Port, UserName, PassWord, VectorName, CollectionName, indexParam, ReplicaNum, dataList, url_split=url_split, return_collection=True
        )
        # 数据质量评估与分流插入
        if status and collection:
            insert_with_quality_check(collection, dataList)
        return status

    Con_status = retry_function(milvus_insert)
    log_event("Milvus插入流程完成")
    return Con_status

def Cre_Search(config, question):
    """
    从配置文件读取参数，执行检索、聚类和重排序。
    """
    milvus_cfg = config["milvus"]
    search_cfg = config.get("search", {})
    embedding_cfg = config["embedding"]

    VectorName = milvus_cfg["vector_name"]
    CollectionName = milvus_cfg["collection_name"]
    topK = search_cfg.get("topK", 10)
    ColChoice = search_cfg.get("col_choice", "hdbscan")
    api_key = embedding_cfg["api_key"]
    reorder_strategy = search_cfg.get("reorder_strategy", "distance")

    log_event(f"开始检索: {question}")
    # 搜索数据
    responseList = search(VectorName, CollectionName, question, topK, api_key)

    # 检查搜索结果是否为空
    if not responseList:
        return {"message": "No results found", "clusters": []}

    # 提取搜索结果中的向量和 ID
    embeddings = [result["embedding"] for result in responseList]
    ids = [result["id"] for result in responseList]

    # 转换为 NumPy 数组
    embeddings = np.array(embeddings)

    # 根据选择的聚类算法进行聚类
    if ColChoice.lower() == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)
    elif ColChoice.lower() == "kmeans":
        k = min(len(embeddings), 5)  # 设置聚类数量，最多为5
        clusterer = KMeans(n_clusters=k, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {ColChoice}")

    # 将聚类结果与搜索结果结合
    clustered_results = {}
    for idx, label in enumerate(labels):
        if label not in clustered_results:
            clustered_results[label] = []
        clustered_results[label].append({
            "id": ids[idx],
            "embedding": embeddings[idx].tolist(),
            "distance": responseList[idx]["distance"]
        })

    # 对聚类结果进行重排序
    query_vector = responseList[0]["embedding"]  # 假设第一个结果的向量为查询向量
    sorted_clusters = reorder_clusters(clustered_results, query_vector, strategy=reorder_strategy)
    log_event(f"检索完成，返回结果数：{len(responseList)}")

    # 返回重排序后的聚类结果
    return {
        "message": "Search, clustering, and reordering completed",
        "clusters": sorted_clusters
    }