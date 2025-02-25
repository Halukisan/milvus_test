import numpy as np
from milvus_insert import insert
from zhipuai import ZhipuAI
import re
import os
api_key = ""
if not api_key:
    raise ValueError("Environment variable ZHIPUAI_API_KEY is not set.")
java_doc = []
# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)
def custom_text_splitter(text):
    # 使用正则表达式匹配句号
    sentence_end_pattern = r'(?<=[。！？])'
    
    # 初始化结果列表和当前段落起始位置
    result = []
    start = 0
    
    # 遍历文本并按照规则分割
    for match in re.finditer(sentence_end_pattern, text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:  # 忽略空字符串
            result.append(segment)
        start = end
    
    # 添加最后一个段落（如果有）
    final_segment = text[start:].strip()
    if final_segment:
        result.append(final_segment)
    
    return result

# 计算两个向量之间的余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 定义处理器抽象基类
class Handler:
    def __init__(self, next_handler=None):
        self.next_handler = next_handler
    
    def handle(self, vector):
        pass

# 生活类处理器
class JAVA_CategoryHandler(Handler):
    def __init__(self, category_vector, threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.category_vector = category_vector
        self.threshold = threshold
    
    def handle(self, vector):
        similarity = cosine_similarity(vector, self.category_vector)
        if similarity >= self.threshold:
            print(f"Vector placed in 'java' category with similarity {similarity:.2f}")
            # 说明这个数据属于java类的，那么开始插入
            return 'java'
        elif self.next_handler is not None:
            return self.next_handler.handle(vector)

# 数学类处理器
class MYSQL_CategoryHandler(Handler):
    def __init__(self, category_vector, threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.category_vector = category_vector
        self.threshold = threshold
    
    def handle(self, vector):
        similarity = cosine_similarity(vector, self.category_vector)
        if similarity >= self.threshold:
            print(f"Vector placed in 'mysql' category with similarity {similarity:.2f}")
            return 'mysql'
        elif self.next_handler is not None:
            return self.next_handler.handle(vector)
        
class REDIS_LCategoryHandler(Handler):   

    def __init__(self, category_vector, threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.category_vector = category_vector
        self.threshold = threshold

    def handle(self, vector):
        similarity = cosine_similarity(vector, self.category_vector)
        if similarity >= self.threshold:
            print(f"Vector placed in 'redis' category with similarity {similarity:.2f}")
            return 'redis'
        elif self.next_handler is not None:
            return self.next_handler.handle(vector)

class WEB_CategoryHandler(Handler):
    def __init__(self, category_vector, threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.category_vector = category_vector
        self.threshold = threshold

    def handle(self, vector):
        similarity = cosine_similarity(vector, self.category_vector)
        if similarity >= self.threshold:
            print(f"Vector placed in 'web' category with similarity {similarity:.2f}")
            return 'web'
        elif self.next_handler is None:
            # 后面没有节点了，返回default，说明插入默认块
            return 'default'


def handle_line(keyVector_java,keyVector_mysql,keyVector_redis,keyVector_web):
    # 创建责任链
    java_category_vector = np.array(keyVector_java)     # java类责任链
    mysql_category_vector = np.array(keyVector_mysql)       # mysql类责任链
    redis_category_vector = np.array(keyVector_redis)        # redis类责任链
    web_category_vector = np.array(keyVector_web)       # 网络类责任链

    handler_java = JAVA_CategoryHandler(category_vector=java_category_vector, next_handler=handler_mysql)
    handler_mysql = MYSQL_CategoryHandler(category_vector=mysql_category_vector, next_handler=handler_redis)
    handler_redis = REDIS_LCategoryHandler(category_vector=redis_category_vector, next_handler=handler_web)
    handler_web = WEB_CategoryHandler(category_vector=web_category_vector)

    # 指定要读取的 Markdown 文件路径
    markdown_file_path = "../data_base/knowledge_db/video_Test_Data"
    # 列出目录中的所有文件
    files_in_directory = os.listdir(markdown_file_path)
    fin_content = ""    
    # 过滤出所有的Markdown文件
    markdown_files = [file for file in files_in_directory if file.endswith('.md')]
    # 读取每个Markdown文件的内容
    for markdown_file in markdown_files:
        file_path = os.path.join(markdown_file_path, markdown_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            segments = custom_text_splitter(content)
            for i, segment in enumerate(segments, 1):
                non_url_text = segment
                non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()  # 清理多余空白
                response = client.embeddings.create(
                    model="embedding-2",  # 替换为您想要使用的模型名称
                    input=non_url_text  # 传递单个文本
                )
                doc_entry = {
                    "id": i,  # 可以根据实际情况调整ID生成逻辑
                    "description": non_url_text,
                    "question_id": i,  # 后面根据业务调整
                    "embedding": response.data[0].embedding 
                }

                # 获取生成的 embedding 值
                vector_to_classify = response.data[0].embedding
                # 分类过程(责任链)
                category = handler_java.handle(vector_to_classify)
                
                insert(category,doc_entry)

                

