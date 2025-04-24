from System.monitor import log_event


def milvus_search(milvus_collection, embedding, topK):
    try:
        milvus_results = milvus_collection.search(
            data=[embedding],
            anns_field="embedding",
            limit=topK,
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            output_fields=["id", "content", "embedding"]
        )
        return [
            {
                "id": hit.id,
                "content": hit.entity.get("content", ""),
                "embedding": hit.entity.get("embedding", []),
                "distance": hit.distance
            }
            for hit in milvus_results[0]
        ]
        
    except Exception as e:
        log_event(f"ES检索异常: {e}")
        return []