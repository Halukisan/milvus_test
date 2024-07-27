import json
from pymilvus import Collection, db, connections
import numpy as np
from zhipuai import ZhipuAI
from pymilvus import MilvusClient

# 替换为您的实际 API 密钥
api_key = "8fb21e517a965b10cf87b7fdadf18a74.UVLbxTJXdL3YLxcT"

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)
# 这里text的内容即为我们需要从向量数据库中检索的内容，通常为用户的问题
text  = ["大语言模型具有什么能力"]
conn = connections.connect(host="127.0.0.1", port=19530)
db.using_database("sample_db")
coll_name = 'word_vector'
# 加载集合，不加载就不能插入数据
collection = Collection(coll_name)
collection.load()
mids, embedings, counts, descs = [], [], [], []
idx = 1
# 调用嵌入创建 API
response = client.embeddings.create(
    model="embedding-2",  # 替换为您想要使用的模型名称
    input=text  # 传递单个文本
)
# print([response.data[0].embedding])
collection = Collection(coll_name)
# 1. 设置一个Milvus客户端
search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
# 确保集合已经被加载并且不是空的
print(collection.is_empty)
# 如果没有执行查询，检查一下collection是否存在，如果存在还不能查询，就重启一下
if not collection.is_empty:
    # 构建查询参数
    print("开始查询")
    res = collection.search(
        anns_field="embeding",
        # 搜索的字段，这里假设有一个文本字段叫做 description
        data=[response.data[0].embedding],
        # 返回 TopK 个最相关的结果
        limit=1,
        param={"metric_type": "IP", "params": {}}, # 搜索参数
        # 下面的desc非常重要，如果没有下面这一句，则不会输出desc的内容。
        output_fields=["desc"] # 返回的输出字段
    )
    print(res)
    

