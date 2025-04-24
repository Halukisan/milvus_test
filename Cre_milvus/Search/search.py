from pymilvus import Collection
from Search.ES.EsSer import es_search
from Search.embedding import EmbeddingGenerator
from System.init import es_client, redis_client
from System.monitor import log_event
from Search.redisSer import search_similar_embedding, cache_embedding_result
from Search.milvusSer import milvus_search


def search(CollectionName, question, topK):
    results = None
    def _do_search():
        log_event("多路召回开始")
        collection = Collection(CollectionName)
        embber = EmbeddingGenerator()
        embedding = embber.get_embedding(question)
        # 2. 先查redis向量缓存
        results = search_similar_embedding(redis_client,embedding,redis_client)
        if results:
            return results
        
        milvus_list = milvus_search(collection, embedding, topK)
        es_list = []
        if es_client:
            # 新增：关键词提取后ES检索
            es_list = es_search(es_client, question, topK, es_index="doc_index")
        if milvus_list:
            log_event("milvus_list结果为空")
        if es_list:
            log_event("es_list结果为空")
        all_results = milvus_list + es_list
        all_results = [r for r in all_results if r.get("embedding")]
        # 4. 写入redis向量缓存
        cache_embedding_result(redis_client, all_results)
        return all_results
    
    results = _do_search()
    return results
