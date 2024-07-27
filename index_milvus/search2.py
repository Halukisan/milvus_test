from pymilvus import MilvusClient, DataType
# 5. Insert more data into the collection
# 5.1. Prepare data
# 1. 设置一个Milvus客户端
client = MilvusClient(
    uri="http://localhost:19530"
)
# 7.搜索多个向量
# 7.1.准备查询向量
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648],
    [0.0039737443, 0.003020432, -0.0006188639, 0.03913546, -0.00089768134]
]
 
# 7.2.开始搜索
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    limit=3,
)
 
print(res)
 
# 输出结果
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
#     ],
#     [
#         {
#             "id": 532,
#             "distance": 0.044551681727170944,
#             "entity": {}
#         },
#         {
#             "id": 149,
#             "distance": 0.044386886060237885,
#             "entity": {}
#         },
#         {
#             "id": 271,
#             "distance": 0.0442606583237648,
#             "entity": {}
#         }
#     ]
# ]
 