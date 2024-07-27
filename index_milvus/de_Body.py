from aheader import client

# 13. 通过ID删除实体
res = client.delete(
    collection_name="quick_setup",
    ids=[0,1,2,3,4]
)
 
print(res)
 
# 输出
#
# {
#     "delete_count": 5
# }

# 14. 通过过滤器表达式删除实体
res = client.delete(
    collection_name="quick_setup",
    filter="id in [5,6,7,8,9]"
)
 
print(res)
 
# 输出
#
# {
#     "delete_count": 5
# }