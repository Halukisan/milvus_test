
# 创建索引
from pymilvus import Collection, utility, connections, db
 
conn = connections.connect(host="127.0.0.1", port=19530)
db.using_database("sample_db")
#  定义索引参数字典，以便对数据进行更高效的检索，，包括了指定的度量类型（metric_type）、索引类型（index_type）和参数（params），使用IP（内积距离）作为度量单位
# IVF_FLAT作为索引类型，并设置了nlist为1024
index_params = {
  "metric_type": "IP",
  "index_type": "IVF_FLAT",
  "params": {"nlist": 1024}
}
# 在倒排列表的索引结构中，nlist参数表示在该索引结构的聚类中心数量，在该索引中，数据会被分为多个聚类中心，这影响了索引的精度和速度，较大的
# nlist会提高搜索的准确性，但也会增加存储开销 
 
# 创建集合对象，milvus中，集合是用来存储向量数据的容器，每个集合可以包含多个向量，可以在这个集合里面进行增加数据，创建索引，查询等。
collection = Collection("word_vector")
collection.create_index(
  field_name="embeding",
  index_params=index_params
)
# 是在集合对象 collection 中的名为 "embeding" 的字段上创建一个索引，索引的类型和参数由 index_params 字典指定。
 
# 检查名为 "word_vector" 的集合中索引的构建进度。
utility.index_building_progress("word_vector")

#  FLAT：准确率高， 适合数据量小，暴力求解相似。
#  IVF-FLAT：量化操作， 准确率和速度的平衡
#  IVF: inverted file 先对空间的点进行聚类，查询时先比较聚类中心距离，再找到最近的N个点。
#  IVF-SQ8：量化操作，disk cpu GPU 友好
#  SQ8：对向量做标量量化，浮点数表示转为int型表示，4字节->1字节。
#  IVF-PQ：快速，但是准确率降低，把向量切分成m段，对每段进行聚类；查询时，查询向量分端后与聚类中心计算距离，各段相加后即为最终距离。使用对称距离(聚类中心之前的距离)不需要计算直接查表，但是误差回更大一些。
#  HNSW：基于图的索引，高效搜索场景，构建多层的NSW。
#  ANNOY：基于树的索引，高召回率
