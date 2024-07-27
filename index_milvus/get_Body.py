from aheader import client
# 12. 按ID获取实体
res = client.get(
    collection_name="quick_setup",
    ids=[1,2,3],
    output_fields=["title", "vector"]
)
 
print(res)
 
# 输出
#
# [
#     {
#         "id": 1,
#         "vector": [
#             0.19886813,
#             0.060235605,
#             0.6976963,
#             0.26144746,
#             0.8387295
#         ]
#     },
#     {
#         "id": 2,
#         "vector": [
#             0.43742132,
#             -0.55975026,
#             0.6457888,
#             0.7894059,
#             0.20785794
#         ]
#     },
#     {
#         "id": 3,
#         "vector": [
#             0.3172005,
#             0.97190446,
#             -0.36981148,
#             -0.48608947,
#             0.9579189
#         ]
#     }
# ]