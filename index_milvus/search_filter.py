from aheader import client
# 8. 使用架构定义的字段进行带过滤表达式的搜索
# 1 准备查询向量
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]
 
# 2. 开始搜索
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    filter="500 < id < 800",
    limit=3
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
#             "id": 505,
#             "distance": 0.0749310627579689,
#             "entity": {}
#         }
#     ]
# ]

# 9. 使用自定义字段进行带过滤表达式的搜索
# 9.1. 准备查询向量
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]
 
# 9.2. 开始搜索
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    filter='$meta["color"] like "red%"',
    limit=3,
    output_fields=["color"]
)
 
print(res)
 
# 输出
#
# [
#     [
#         {
#             "id": 240,
#             "distance": 0.0694073885679245,
#             "entity": {
#                 "color": "red_8667"
#             }
#         },
#         {
#             "id": 581,
#             "distance": 0.059804242104291916,
#             "entity": {
#                 "color": "red_1786"
#             }
#         },
#         {
#             "id": 372,
#             "distance": 0.049707964062690735,
#             "entity": {
#                 "color": "red_2186"
#             }
#         }
#     ]
# ]
 