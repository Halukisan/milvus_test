import random
from pymilvus import MilvusClient, DataType
# 5. Insert more data into the collection
# 5.1. Prepare data
# 1. 设置一个Milvus客户端
client = MilvusClient(
    uri="http://localhost:19530"
)
# 以下内容用于插入更多的数据，以体现milvus的搜索能力
colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
data = [ {
    "id": i, 
    "vector": [ random.uniform(-1, 1) for _ in range(5) ], 
    "color": f"{random.choice(colors)}_{str(random.randint(1000, 9999))}" 
} for i in range(1000) ]
 
# 5.2. Insert data
res = client.insert(
    collection_name="quick_setup",
    data=data[10:]
)
 
print(res)
 
# Output
#
# {
#     "insert_count": 990
# }