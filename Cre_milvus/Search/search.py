from pymilvus import Collection
from Search.EsSer import es_search
from Search.redisSer import cached_search
from zhipuai import ZhipuAI
from System.init import es_client, redis_client
from elasticsearch import Elasticsearch
from .keyword_extractor import KeywordExtractor

def search(VectorName, CollectionName, question, topK, api_key):
    query_key = f"{VectorName}_{CollectionName}_{question}_{topK}"

    def _do_search():
        collection = Collection(CollectionName)
        client = ZhipuAI(api_key=api_key)
        embedding = client.embeddings.create(model="embedding-2", input=[question]).data[0].embedding
        milvus_results = collection.search(
            data=[embedding],
            anns_field="embedding",
            limit=topK,
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            output_fields=["id", "content", "embedding"]
        )
        milvus_list = [
            {
                "id": hit.id,
                "content": hit.entity.get("content", ""),
                "embedding": hit.entity.get("embedding", []),
                "distance": hit.distance
            }
            for hit in milvus_results[0]
        ]
        es_list = []
        if es_client:
            # 新增：关键词提取后ES检索
            extractor = KeywordExtractor()
            keywords = extractor.extract_keywords(question, top_k=5)
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": kw}} for kw in keywords
                        ]
                    }
                }
            }
            resp = es_client.search(index="doc_index", body=es_query, size=topK)
            es_list = [
                {
                    "id": hit["_id"],
                    "content": hit["_source"].get("content", ""),
                    "embedding": hit["_source"].get("embedding", []),
                    "distance": hit["_score"]
                }
                for hit in resp["hits"]["hits"]
            ]
        all_results = milvus_list + es_list
        all_results = [r for r in all_results if r.get("embedding")]
        return all_results

    results = cached_search(query_key, _do_search, redis_client)
    return results

def search_by_keywords(query, top_k=5, es_host="http://localhost:9200", index_name="doc_index"):
    es_client = Elasticsearch(es_host)
    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords(query, top_k=top_k)
    es_query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"content": kw}} for kw in keywords
                ]
            }
        }
    }
    res = es_client.search(index=index_name, body=es_query)
    return res["hits"]["hits"], keywords