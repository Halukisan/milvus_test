from aheader import client
# 10. 使用已定义字段进行过滤的查询表达式

res = client.query(
    collection_name="quick_setup",
    filter="10 < id < 15",
    output_fields=["color"]
)
 
print(res)
 
# 输出
#
# [
#     {
#         "color": "green_7413",
#         "id": 11
#     },
#     {
#         "color": "orange_1417",
#         "id": 12
#     },
#     {
#         "color": "orange_6143",
#         "id": 13
#     },
#     {
#         "color": "white_4084",
#         "id": 14
#     }
# ]
 