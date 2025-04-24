from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import hdbscan
import logging
from System.eval import insert_with_quality_check
def milvus_connect_insert(CollectionName, IndexParam, ReplicaNum, dataList, url_split, return_collection=False):
    try:
        # 不再重复连接Milvus，连接已在init.py完成

        # 检查并创建collection
        collection_name = CollectionName
        if url_split:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="cluster_label", dtype=DataType.INT64)
            ]
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="cluster_label", dtype=DataType.INT64)
            ]
        schema = CollectionSchema(fields, collection_name)
        collection = Collection(name=collection_name, schema=schema)

        # 聚类
        embeddings = [data["embedding"] for data in dataList]
        clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)

        # 插入数据
        for i, data in enumerate(dataList):
            data["cluster_label"] = labels[i]

        # 评估数据质量,高质量的数据会自动插入milvus，低质量的数据将会插入到low_quality_data集合中
        hig_dataList_insert_status,low_dataList_insert_status = insert_with_quality_check(dataList) 
        if not low_dataList_insert_status:
            # 低质量数据插入失败，程序继续运行
            logging.error("Low quality data insertion failed.")
        if not hig_dataList_insert_status:
            # 高质量数据插入失败，程序结束运行
            logging.error("High quality data insertion failed.")
            return False, None

        
        # 创建索引
        collection.create_index(field_name="embedding", index_params=IndexParam)

        # 启用内存副本
        collection.load(replica_number=ReplicaNum)
        if return_collection:
            return True, collection
        else:
            return True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if return_collection:
            return False, None
        else:
            return False