from elasticsearch import Elasticsearch, helpers

def create_index(es_client, index_name):
    mapping = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "file_name": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "meta": {"type": "object", "enabled": True},
                "embedding": {"type": "dense_vector", "dims": 768}
            }
        }
    }
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mapping)

def bulk_insert_to_es(es_client, index_name, data_list):
    actions = [
        {"_index": index_name, "_source": doc}
        for doc in data_list
    ]
    helpers.bulk(es_client, actions)