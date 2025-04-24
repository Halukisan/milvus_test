import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Search.embedding import EmbeddingGenerator

def process_md(md_dir_path, url_split):
    """
    处理 Markdown 文件并返回一个包含 id、content、embedding 和 URL 的列表。

    参数:
        md_dir_path (str): Markdown 文件所在目录路径。
        model_name (str): 嵌入模型名称。
        api_key (str): ZhipuAI 的 API 密钥。

    返回:
        list: 包含 id、content、embedding 和 URL 的列表，格式为 [{'id': id, 'content': content, 'embedding': embedding, 'urls': [url1, url2, ...]}, ...]。
    """

    # 定义正则表达式清洗函数
    def clean_content(content):
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content)
        content = content.replace('•', '').replace(' ', '').replace('*', '').replace('\n\n', '\n')
        return content

    # 提取 URL 的函数
    def extract_urls_with_positions(text):
        url_pattern = r'(https?://[^\s\)\]\}>]+)'
        return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]

    # 定义文本分割器
    CHUNK_SIZE = 500
    OVERLAP_SIZE = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    # 列出目录中的所有 Markdown 文件
    if not os.path.exists(md_dir_path):
        raise FileNotFoundError(f"目录 {md_dir_path} 不存在")
    markdown_files = [file for file in os.listdir(md_dir_path) if file.endswith('.md')]

    results = []
    current_id = 1

    # 处理每个 Markdown 文件
    for markdown_file in markdown_files:
        file_path = os.path.join(md_dir_path, markdown_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 清洗内容
        cleaned_content = clean_content(content)

        # 分割内容
        split_docs = text_splitter.split_text(cleaned_content)

        # 为每段内容生成嵌入
        for doc in split_docs:
            urls_with_positions = extract_urls_with_positions(doc)
            non_url_text = doc

            # 移除 URL 并清理多余空白
            for url, (start, end) in reversed(urls_with_positions):
                non_url_text = non_url_text[:start] + non_url_text[end:]
            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()

            embedder = EmbeddingGenerator()
            
            embedding = embedder.get_embedding(non_url_text)
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