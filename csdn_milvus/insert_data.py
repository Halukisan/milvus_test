from pymilvus import Collection, db, connections
import numpy as np
 
conn = connections.connect(host="127.0.0.1", port=19530)
db.using_database("sample_db")
coll_name = 'word_vector'
# 加载集合，不加载就不能插入数据
collection = Collection(coll_name)
collection.load()
mids, embedings, counts, descs = [], [], [], []
data_num = 100
for idx in range(0, data_num):
    mids.append(idx)
    embedings.append(np.random.normal(0, 0.1, 768).tolist())
    descs.append(f'random num {idx}')
    counts.append(idx)

# 创建一个叫word_vector的集合对象
collection = Collection(coll_name)
# 调用insert（）方法，将数据插入到集合中
mr = collection.insert([mids, embedings, descs, counts])
print(mr)