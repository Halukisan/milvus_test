import json

def cached_search(query_key, search_function, redis_client):
    try:
        cached = redis_client.get(query_key)
        if cached:
            return json.loads(cached)
        results = search_function()
        redis_client.set(query_key, json.dumps(results), ex=3600)
        return results
    except Exception as e:
        from System.monitor import log_event
        log_event(f"Redis缓存异常: {e}")
        return search_function()