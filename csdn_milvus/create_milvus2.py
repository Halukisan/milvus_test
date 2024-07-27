from pymilvus import connections, db
# 创建数据库
conn = connections.connect(host="127.0.0.1", port=19530)
# 注意创建数据库时，去attu里面看一看有没有创建出来，attu可能卡住的，退出再重新连接就可以了，直到显示出你新创建的数据库为止，才可以创建索引
# database = db.create_database("sample_db")

# 切换到sample_db数据库
db.using_database("sample_db")

# 列出所有数据库
# db.list_database()
# print(db.list_database())

from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections

# 集合名称
collection_name = "word_vector"
# 删除集合
# try:
#     collection = Collection(name=collection_name)
    
#     collection.drop()
#     print(f"集合 {collection_name} 已被删除。")
# except Exception as e:
#     print(f"删除集合时发生错误：{e}")

# 创建行
m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True,auto_id = True)
embeding = FieldSchema(name="embeding", dtype=DataType.FLOAT_VECTOR, dim=1024,)
# count = FieldSchema(name="count", dtype=DataType.INT64,)
desc = FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=2048,)

schema = CollectionSchema(
  # fields=[m_id, embeding, desc, count],
  fields=[m_id,embeding, desc],

  description="Test embeding search",
  enable_dynamic_field=True
)
 
collection_name = "word_vector"

collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)











