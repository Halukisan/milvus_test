import re
import os
import time
from urllib.parse import urlparse

def is_valid_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_urls_with_positions(text):
    url_pattern = r'(https?://[^\s\)\]\}>]+)'
    return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]

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
            urls_with_positions = extract_urls_with_positions(segment)
            non_url_text = segment
            for url, (start, end) in reversed(urls_with_positions):
                # 从后往前替换URL，避免索引偏移
                non_url_text = non_url_text[:start] + non_url_text[end:]

            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()  # 清理多余空白
            print(f"第{i}部分开始-----------------------------------------------------------------------------------------")
            print(f"Segment {i} Text: {non_url_text}")
            for url, _ in urls_with_positions:
                print(f"URL: {url}")

            #time.sleep(4)  # 等待4秒