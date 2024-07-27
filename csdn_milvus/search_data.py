from pymilvus import Collection, db, connections
import numpy as np
 
conn = connections.connect(host="127.0.0.1", port=19530)
db.using_database("sample_db")
coll_name = 'word_vector'
 
search_params = {
    "metric_type": 'IP',
    "offset": 0,
    "ignore_growing": False,
    "params": {"nprobe": 16}
}
 
# 上一步插入数据的时候已经加载过啦
collection = Collection(coll_name)
# collection.load()
 
results = collection.search(
    data=[np.random.normal(0, 0.1, 768).tolist()],
    anns_field="embeding",
    param=search_params,
    limit=16,
    expr=None,
    # output_fields=['m_id', 'embeding', 'desc', 'count'],
    output_fields=['m_id', 'desc', 'count'],
    consistency_level="Strong"
)
collection.release()
print(results[0].ids)
print(results[0].distances)
hit = results[0][0]
print(hit.entity.get('desc'))
print(results)
 