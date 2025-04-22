from elasticsearch import Elasticsearch
def es_search(es_client, question, topK, es_index="your_index"):
    try:
        resp = es_client.search(
            index=es_index,
            body={"query": {"match": {"content": question}}, "size": topK}
        )
        return [
            {
                "id": hit["_id"],
                "content": hit["_source"].get("content", ""),
                "embedding": hit["_source"].get("embedding", []),
                "distance": hit["_score"]
            }
            for hit in resp["hits"]["hits"]
        ]
    except Exception as e:
        from System.monitor import log_event
        log_event(f"ES检索异常: {e}")
        return []