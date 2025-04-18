import csv
import numpy as np
from towhee import pipe, ops
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from milvus_utils import process_and_store_csv_to_milvus

# 调用方法处理 CSV 文件并存储到 Milvus
process_and_store_csv_to_milvus(
    csv_path='reverse_image_search.csv',
    collection_name='text_image_search',
    model_name='clip-vit-base-patch16',
    dim=512,
    host='127.0.0.1',
    port='19530'
)


def process_and_store_csv_to_milvus(csv_path, collection_name, model_name='model', dim=512, host='127.0.0.1', port='19530'):
    """
    处理 CSV 文件中的图片数据并存储到 Milvus 中。

    参数:
        csv_path (str): CSV 文件路径。
        collection_name (str): Milvus 集合名称。
        model_name (str): 用于向量化的模型名称。
        dim (int): 向量维度。
        host (str): Milvus 服务主机地址。
        port (str): Milvus 服务端口。
    """
    # 创建 Milvus 集合
    def create_milvus_collection(collection_name, dim):
        connections.connect("default", host=host, port=port)

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, description='ids', is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # 为集合创建 IVF_FLAT 索引
        index_params = {
            'metric_type': 'L2',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 512}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection

    collection = create_milvus_collection(collection_name, dim)

    # 定义 CSV 文件读取逻辑
    def read_csv(csv_path, encoding='utf-8-sig'):
        with open(csv_path, 'r', encoding=encoding) as f:
            data = csv.DictReader(f)
            for line in data:
                yield int(line['id']), line['path']

    # 定义处理管道
    p3 = (
        pipe.input('csv_file')
        .flat_map('csv_file', ('id', 'path'), read_csv)
        .map('path', 'img', ops.image_decode.cv2('rgb'))
        .map('img', 'vec', ops.image_text_embedding.clip(model_name=model_name, modality='image', device=0))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
        .map(('id', 'vec'), (), ops.ann_insert.milvus_client(host=host, port=port, collection_name=collection_name))
        .output()
    )

    # 执行管道
    p3(csv_path)

    # 加载集合并打印插入数据数量
    collection.load()
    print(f'Total number of inserted data is {collection.num_entities}.')