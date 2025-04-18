import csv
import numpy as np
from towhee import pipe, ops
import os
def process_csv(csv_path, model_name, api_key):
    """
    处理 CSV 文件中的图片数据并返回图片 ID 和向量数据的列表。

    参数:
        csv_path (str): CSV 文件路径。
        model_name (str): 用于向量化的模型名称。

    返回:
        list: 包含图片 ID 和向量数据的列表，格式为 [(id, vector), ...]。
    """
    # 定义 CSV 文件读取逻辑
    def read_csv(csv_path, encoding='utf-8'):
            # 列出目录中的所有 Markdown 文件
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"目录 {csv_path} 不存在")
        csv_files = [file for file in os.listdir(csv_path) if file.endswith('.md')]
        for csv_path in csv_files:
            with open(csv_path, 'r', encoding=encoding) as f:
                data = csv.DictReader(f)
                for line in data:
                    yield int(line['id']), line['path']

    # 定义处理管道
    results = []

    def collect_results(id, vec):
        results.append((id, vec))

    p3 = (
        pipe.input('csv_file')
        .flat_map('csv_file', ('id', 'path'), read_csv)
        .map('path', 'img', ops.image_decode.cv2('rgb'))
        .map('img', 'vec', ops.image_text_embedding.clip(model_name=model_name, modality='image', device=0))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
        .map(('id', 'vec'), (), collect_results)  # 收集结果到列表
        .output()
    )

    # 执行管道
    p3(csv_path)

    return results