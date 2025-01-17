import pymilvus
print(pymilvus.__version__)

from pymilvus import connections,db, FieldSchema, CollectionSchema, DataType, Collection

# 连接Milvus服务器
connections.connect("default", host="localhost", port="19530")
index_params = {
  "metric_type": "IP",
  "index_type": "IVF_FLAT",
  "params": {"nlist": 1024}
}
# 定义collection的schema
fields = [
    FieldSchema(name="video_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="question_id", dtype=DataType.INT64),
    FieldSchema(name="embedding",dtype=DataType.FLOAT_VECTOR,dim=1024)
]
schema = CollectionSchema(fields, "video_push")

# 创建collection
collection_name = 'video_push'
collection = Collection(name=collection_name, schema=schema)
collection = Collection(collection_name)
collection.create_index(
  field_name="embedding",
  index_params=index_params
)
collection.load()

from zhipuai import ZhipuAI

# 替换为您的实际 API 密钥
api_key = ""

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)

videos = [
    {
        "video_id": 1000001,
        "url": "https://example.com/video/1000001",
        "title": "RAG模型介绍",
        "description": "了解RAG模型的基本概念和它在信息检索中的应用。",
        "question_id": 2000001,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000002,
        "url": "https://example.com/video/1000002",
        "title": "视频推送技术基础",
        "description": "视频推送技术的基本原理和实践应用。",
        "question_id": 2000002,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000003,
        "url": "https://example.com/video/1000003",
        "title": "如何实现视频问答系统",
        "description": "构建一个基于RAG的视频问答系统的详细步骤。",
        "question_id": 2000003,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000004,
        "url": "https://example.com/video/1000004",
        "title": "RAG模型在推荐系统中的应用",
        "description": "探讨RAG模型在推荐系统中的实际应用案例。",
        "question_id": 2000004,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000005,
        "url": "https://example.com/video/1000005",
        "title": "视频内容匹配算法",
        "description": "介绍视频内容匹配算法，以及如何与RAG模型结合。",
        "question_id": 2000005,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000006,
        "url": "https://example.com/video/1000006",
        "title": "RAG模型优化技巧",
        "description": "分享RAG模型性能优化的几种常用技巧。",
        "question_id": 2000006,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000007,
        "url": "https://example.com/video/1000007",
        "title": "视频推荐系统设计",
        "description": "设计一个高效的视频推荐系统的基本框架。",
        "question_id": 2000007,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000008,
        "url": "https://example.com/video/1000008",
        "title": "RAG模型与NLP技术",
        "description": "探讨RAG模型与自然语言处理技术的结合。",
        "question_id": 2000008,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000009,
        "url": "https://example.com/video/1000009",
        "title": "视频数据分析",
        "description": "分析视频数据以优化推荐系统。",
        "question_id": 2000009,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000010,
        "url": "https://example.com/video/1000010",
        "title": "RAG模型在视频搜索中的应用",
        "description": "RAG模型如何提升视频搜索的准确性和相关性。",
        "question_id": 2000010,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000011,
        "url": "https://example.com/video/1000011",
        "title": "视频推荐系统算法",
        "description": "介绍视频推荐系统中使用的各种算法。",
        "question_id": 2000011,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000012,
        "url": "https://example.com/video/1000012",
        "title": "RAG模型训练",
        "description": "详细讲解RAG模型的训练过程。",
        "question_id": 2000012,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000013,
        "url": "https://example.com/video/1000013",
        "title": "视频内容识别技术",
        "description": "探讨视频内容识别技术及其在推荐系统中的应用。",
        "question_id": 2000013,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000014,
        "url": "https://example.com/video/1000014",
        "title": "RAG模型与深度学习",
        "description": "分析RAG模型与深度学习技术的关联。",
        "question_id": 2000014,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000015,
        "url": "https://example.com/video/1000015",
        "title": "视频推荐系统评估",
        "description": "如何评估视频推荐系统的性能。",
        "question_id": 2000015,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000016,
        "url": "https://example.com/video/1000016",
        "title": "RAG模型在视频分类中的应用",
        "description": "探讨RAG模型在视频分类任务中的表现和技巧。",
        "question_id": 2000016,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000017,
        "url": "https://example.com/video/1000017",
        "title": "视频推荐算法的比较研究",
        "description": "比较不同视频推荐算法的性能和适用场景。",
        "question_id": 2000017,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000018,
        "url": "https://example.com/video/1000018",
        "title": "RAG模型在视频摘要生成中的作用",
        "description": "分析RAG模型在自动生成视频摘要中的关键作用。",
        "question_id": 2000018,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000019,
        "url": "https://example.com/video/1000019",
        "title": "视频内容推荐系统的挑战",
        "description": "讨论构建视频内容推荐系统时面临的主要挑战。",
        "question_id": 2000019,
        "embedding": [0.0] * 128  # 示例向量数据
    },
    {
        "video_id": 1000020,
        "url": "https://example.com/video/1000020",
        "title": "RAG模型在多模态学习中的应用",
        "description": "研究RAG模型在处理多模态数据时的应用和效果。",
        "question_id": 2000020,
        "embedding": [0.0] * 128  # 示例向量数据
    }
]

for video in videos:
    # 调用嵌入创建 API
    response = client.embeddings.create(
        model="embedding-2",  # 替换为您想要使用的模型名称
        input=video["description"]  # 传递单个文本
    )

    print(f"原文：{video["description"]},response：{response}")
    # 获取生成的 embedding 值
    embedding_value = response.data[0].embedding
    
    # 将生成的 embedding 值替换掉原始的 embedding 值
    video["embedding"] = embedding_value


# 将数据插入Milvus
entities = [
    [video['video_id'] for video in videos],
    [video['url'] for video in videos],
    [video['title'] for video in videos],
    [video['description'] for video in videos],
    [video['question_id'] for video in videos],
    [video['embedding'] for video in videos]  # 添加向量数据
]

mr = collection.insert(entities)
print(mr)
# 关闭Milvus连接
connections.disconnect("default")