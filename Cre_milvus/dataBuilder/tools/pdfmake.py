import os
import re
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai import ZhipuAI
from urllib.parse import urlparse

def process_pdf(pdf_path, model_name, api_key, url_split):
    """
    处理 PDF 文件并返回一个包含 id、content、embedding 和 URL 的列表。

    参数:
        pdf_path (str): PDF 文件路径。
        model_name (str): 嵌入模型名称。
        api_key (str): ZhipuAI 的 API 密钥。

    返回:
        list: 包含 id、content、embedding 和 URL 的列表，格式为 [{'id': id, 'content': content, 'embedding': embedding, 'urls': [url1, url2, ...]}, ...]。
    """
    # 初始化 ZhipuAI 客户端
    client = ZhipuAI(api_key=api_key)

    # 定义正则表达式清洗函数
    def clean_content(content):
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content)
        content = content.replace('•', '').replace(' ', '').replace('\n\n', '\n')
        return content

    # 提取 URL 的函数
    def extract_urls_with_positions(text):
        url_pattern = r'(https?://[^\s\)\]\}>]+)'
        return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]

    # 加载 PDF 文件
    loader = PyMuPDFLoader(pdf_path)
    pdf_pages = loader.load()

    # 定义文本分割器
    CHUNK_SIZE = 500
    OVERLAP_SIZE = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    results = []
    current_id = 1

    # 处理每页内容
    for page in pdf_pages:
        cleaned_content = clean_content(page.page_content)
        split_docs = text_splitter.split_text(cleaned_content)

        # 为每段内容生成嵌入
        for doc in split_docs:
            urls_with_positions = extract_urls_with_positions(doc)
            non_url_text = doc

            # 移除 URL 并清理多余空白
            for url, (start, end) in reversed(urls_with_positions):
                non_url_text = non_url_text[:start] + non_url_text[end:]
            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()

            response = client.embeddings.create(
                model=model_name,
                input=non_url_text
            )
            embedding = response['data'][0]['embedding']
            if url_split:
                results.append({
                    'id': current_id,
                    'content': non_url_text,
                    'embedding': embedding,
                    'urls': [url for url, _ in urls_with_positions]
                })
                current_id += 1
            else:  
                results.append({
                    'id': current_id,
                    'content': non_url_text,
                    'embedding': embedding,
                })
                current_id += 1

    return results