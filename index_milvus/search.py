from pymilvus import MilvusClient, DataType
# 5. Insert more data into the collection
# 5.1. Prepare data
# 1. 设置一个Milvus客户端
client = MilvusClient(
    uri="http://localhost:19530"
)
# 6. 使用单向量进行搜索
# 6.1. 准备查询向量
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]
 
# 6.2. 开始搜索
res = client.search(
    collection_name="quick_setup",     # 目标集合
    data=query_vectors,                # 查询向量
    limit=3,                           # 返回的实体数量
)
 
print(res)
 
# 输出
#
# [
#     [
#         {
#             "id": 548,
#             "distance": 0.08589144051074982,
#             "entity": {}
#         },
#         {
#             "id": 736,
#             "distance": 0.07866684347391129,
#             "entity": {}
#         },
#         {
#             "id": 928,
#             "distance": 0.07650312781333923,
#             "entity": {}
#         }
#     ]
# ]
 