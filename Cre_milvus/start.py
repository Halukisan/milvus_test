from dataBuilder.data import data_process
from milvusBuilder import milvus_connect
from IndexParamBuilder import indexParams
from Search import search
from pymilvus import connections

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

def Cre_Search(VectorName,CollectionName,IP,Port,UserName,PassWord,question,topK,ColChoice,api_key):
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

    返回值：返回搜索结果
    """
    # 连接到Milvus数据库
    connections.connect(VectorName,host=IP, port=Port, user=UserName, password=PassWord)
    # 搜索数据
    responses = search(VectorName,CollectionName,IP,Port,UserName,PassWord,question,topK,api_key)
    # 返回搜索结果列表

    # 选择聚类算法
    




    

