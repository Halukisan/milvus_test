from pymilvus import connections
from elasticsearch import Elasticsearch
import redis

# 全局连接对象
milvus_connected = False
es_client = None
redis_client = None

def init_milvus(alias, host, port, user, password):
    global milvus_connected
    if not milvus_connected:
        connections.connect(alias, host=host, port=port, user=user, password=password)
        milvus_connected = True

def init_es(host):
    global es_client
    if es_client is None:
        es_client = Elasticsearch([host])

def init_redis(host='localhost', port=6379):
    global redis_client
    if redis_client is None:
        redis_client = redis.StrictRedis(host=host, port=port, db=0)