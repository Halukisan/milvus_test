import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import redis
import logging
import pickle
from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 初始化模型和工具
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 日志配置
logging.basicConfig(filename='search_engine.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SearchRequest(BaseModel):
    query: str
    knowledge_base_ids: List[str] = None
    search_strategy: str = 'hybrid'

class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    source: str

# 检索策略实现
def bm25_search(query: str, documents: List[str]) -> List[SearchResult]:
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    results = [SearchResult(id=str(i), content=doc, score=score, source='bm25') 
               for i, (doc, score) in enumerate(zip(documents, scores))]
    return sorted(results, key=lambda x: x.score, reverse=True)

def semantic_search(query: str, documents: List[str]) -> List[SearchResult]:
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(documents)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    results = [SearchResult(id=str(i), content=doc, score=score, source='semantic') 
               for i, (doc, score) in enumerate(zip(documents, similarities))]
    return sorted(results, key=lambda x: x.score, reverse=True)

def hybrid_search(query: str, documents: List[str]) -> List[SearchResult]:
    bm25_results = bm25_search(query, documents)
    semantic_results = semantic_search(query, documents)
    combined_results = {}
    for result in bm25_results + semantic_results:
        if result.id not in combined_results or result.score > combined_results[result.id].score:
            combined_results[result.id] = result
    return sorted(combined_results.values(), key=lambda x: x.score, reverse=True)

# 缓存机制
def get_cached_results(query: str) -> Union[List[SearchResult], None]:
    cached = redis_client.get(f'search:{query}')
    return pickle.loads(cached) if cached else None

def cache_results(query: str, results: List[SearchResult]):
    redis_client.setex(f'search:{query}', 3600, pickle.dumps(results))

# 结果重排序
def rerank_results(results: List[SearchResult], user_feedback: Dict[str, float] = None) -> List[SearchResult]:
    if user_feedback:
        for result in results:
            if result.id in user_feedback:
                result.score *= (1 + user_feedback[result.id])
    return sorted(results, key=lambda x: x.score, reverse=True)

# 聚类功能
def cluster_results(results: List[SearchResult], threshold: float = 0.8) -> List[List[SearchResult]]:
    if not results:
        return []
    embeddings = model.encode([result.content for result in results])
    similarity_matrix = cosine_similarity(embeddings)
    clusters = []
    clustered_indices = set()
    for i in range(len(results)):
        if i not in clustered_indices:
            cluster = [results[i]]
            clustered_indices.add(i)
            for j in range(i+1, len(results)):
                if j not in clustered_indices and similarity_matrix[i][j] >= threshold:
                    cluster.append(results[j])
                    clustered_indices.add(j)
            clusters.append(cluster)
    return clusters

# API端点
@app.post('/search')
def search(request: SearchRequest) -> Dict[str, Union[List[SearchResult], List[List[SearchResult]]]]:
    # 从知识库获取文档（模拟）
    documents = ["文档1内容", "文档2内容", "文档3内容"]  # 实际应从知识库加载
    
    # 检查缓存
    cached = get_cached_results(request.query)
    if cached:
        logging.info(f'Cache hit for query: {request.query}')
        return {'results': cached, 'source': 'cache'}
    
    # 执行检索
    if request.search_strategy == 'bm25':
        results = bm25_search(request.query, documents)
    elif request.search_strategy == 'semantic':
        results = semantic_search(request.query, documents)
    else:
        results = hybrid_search(request.query, documents)
    
    # 重排序和聚类
    reranked = rerank_results(results)
    clustered = cluster_results(reranked)
    
    # 缓存结果
    cache_results(request.query, reranked)
    logging.info(f'Search performed for query: {request.query}')
    
    return {'results': reranked, 'clusters': clustered, 'source': 'search'}

# 单元测试（示例）
def test_bm25_search():
    docs = ["苹果", "香蕉", "苹果手机"]
    results = bm25_search("苹果", docs)
    assert len(results) == 2
    assert results[0].content == "苹果"

def test_semantic_search():
    docs = ["机器学习", "深度学习", "人工智能"]
    results = semantic_search("AI", docs)
    assert len(results) == 3
    assert results[0].content == "人工智能"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)