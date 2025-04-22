import time

def retry_function(func, max_retries=3, delay=2):
    """
    通用重试机制。
    """
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            print(f"Error: {e}, retrying {retries}/{max_retries}...")
            time.sleep(delay)
    raise RuntimeError("Max retries reached")