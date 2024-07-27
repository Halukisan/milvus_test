from pymilvus import Collection, db, connections
 
conn = connections.connect(host="127.0.0.1", port=19530)
db.using_database("sample_db")
coll_name = 'word_vector'
 
collection = Collection(coll_name)
 
ids = [str(idx) for idx in range(10)]
temp_str = ', '.join(ids)
query_expr = f'm_id in [{temp_str}]'
result = collection.delete(query_expr)
 
print(result)