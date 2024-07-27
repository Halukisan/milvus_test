from pymilvus import MilvusClient, DataType

# 1. 设置一个Milvus客户端
client = MilvusClient(
    uri="http://localhost:19530"
)