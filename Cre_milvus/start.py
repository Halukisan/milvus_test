from dataBuilder.data import data_process
from milvusBuilder import milvus_connect
from IndexParamBuilder import indexParams
from reorder import reorder_clusters
from Search import search
from pymilvus import connections
import hdbscan
from sklearn.cluster import KMeans
import numpy as np

def Cre_VectorDataBaseStart(C_G_Choic,IP,Port,UserName,PassWord,VectorName,CollectionName,IndexName,ReplicaNum,Data_Location,Data_Type,url_split,embedding_model,api_key):
    """
    C_G_Choic：选择GPU索引还是CPU索引，cpu gpu
    IP：milvus数据库的地址
    Port：端口
    UserName：用户名
    PassWord：密码
    VectorName：向量数据库名称
    CollectionName：集合名称
    IndexName：索引选择 对于GPU索引选择GPU_BRUTE_FORCE，对于CPU索引选择IVF、IVF_FLAT，HNSW，HNSW_SQ8等
    ReplicaNum：是否启用内存副本 输入内存副本的数量，0表示不启用内存副本
    Data_Location：数据文件的路径pdf、txt、md、csv。图片数据请把图片的地址放到csv文件中。
    Data_Type:文件类型
    url_split:是否启用url切分和提取功能 True or False

    embedding_model：选择embedding模型
    api_key：api_key
    返回值：true or false表示向量数据库构建是否成功
    """

    # 数据处理
        # dataList的数据格式有两种情况，当url_split=True时，数据格式为[{'id': id, 'content': content, 'embedding': embedding, 'urls': [url1, url2, ...]}, ...]，否则数据格式为[(id,content,embedding), ...]。
    dataList = data_process(data_location=Data_Location,data_type=Data_Type,model_name=embedding_model,api_key=api_key,url_split=url_split)
    # 构建索引
    indexParam = indexParams(C_G_Choic,IndexName)
    # 连接向量数据库并构建,插入数据
    Con_status = milvus_connect(IP,Port,UserName,PassWord,VectorName,CollectionName,indexParam,ReplicaNum,dataList,url_split=url_split)
    return Con_status

def Cre_Search(VectorName,CollectionName,IP,Port,UserName,PassWord,question,topK,ColChoice,api_key,reorder_strategy="distance"):
    """
    IP：milvus数据库的地址
    Port：端口
    UserName：用户名
    PassWord：密码
    VectorName：向量数据库名称
    CollectionName：集合名称
    txt：需要搜索的文本
    topK：搜索结果的数量
    ColChoice：聚类算法选择 kmeans hdbscan
    EmbeddingModelName： 选择embedding模型
    modelName：选择问答模型
    api_key：api_key默认用的是ChatGLM
    reorder_strategy (str): 重排序策略，可选值为 "distance", "cluster_size", "cluster_center"。

    返回值：返回搜索结果
    """
    
    # 连接到 Milvus 数据库
    connections.connect(VectorName, host=IP, port=Port, user=UserName, password=PassWord)

    # 搜索数据
    responseList = search(VectorName, CollectionName, IP, Port, UserName, PassWord, question, topK, api_key)

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

    # 返回重排序后的聚类结果
    return {
        "message": "Search, clustering, and reordering completed",
        "clusters": sorted_clusters
    }





    

