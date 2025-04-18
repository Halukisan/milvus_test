from elasticsearch import Elasticsearch


# 连接到Elasticsearch
es_client = Elasticsearch(["http://localhost:9200"])
es_index_name = "your_es_index"


def search_elasticsearch(query_text, top_k=6):
    response = es_client.search(
        index=es_index_name,
        body={
            "size": top_k,
            "query": {
                "match": {
                    "content": query_text
                }
            }
        }
    )
    return [(hit['_id'], hit['_score']) for hit in response['hits']['hits']]

