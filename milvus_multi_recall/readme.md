# 多路召回：
1. Milvus：使用Milvus进行向量搜索，获取最相似的文档。
2. Elasticsearch：使用Elasticsearch进行关键词搜索，获取最相似的文档。
3. function_calling：基于通义千问，联网查询，获得最新数据。
4. 将以上三个结果进行排序，并返回最相似的文档。
5. prompt中限制模型回答，temperature设置为0.1。
6. 接入text_search_pic功能

对于数据存储部分，将所有数据分别都存入milvus和ES中，milvus中存储结合video_url_test中的代码，提供URL识别与拆分，然后采用HDBSCAN聚类算法，采用GPU索引CPU_BRUTE_FORCE

而对于ES部分，采用DFS QUERY THEN　ＦＥＴＣＨ方式，在查询之前先分散计算各分片的词频信息，然后再执行查询，这种方式能提供更加准确的结果。
