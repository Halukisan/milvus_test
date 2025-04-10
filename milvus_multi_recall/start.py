from pymilvus import connections, Collection
from elasticsearch import Elasticsearch
from ESPart.EsSearch import search_elasticsearch
from milvusPart.milvusSearch import get_embedding
from functioncall.FunctionTools import call_with_messages
import gradio as gr
import pandas as pd
import cv2
from dashscope import Generation
from towhee import ops, pipe, DataCollection
import numpy as np
"""
多路召回：
1. Milvus：使用Milvus进行向量搜索，获取最相似的文档。
2. Elasticsearch：使用Elasticsearch进行关键词搜索，获取最相似的文档。
3. function_calling：基于通义千问，联网查询，获得最新数据。
4. 将以上三个结果进行排序，并返回最相似的文档。
5. prompt中限制模型回答，temperature设置为0.1。
6. 接入text_search_pic功能

对于数据存储部分，将所有数据分别都存入milvus和ES中，milvus中存储结合video_url_test中的代码，提供URL识别与拆分，然后采用HDBSCAN聚类算法，采用GPU索引CPU_BRUTE_FORCE
而对于ES部分，采用DFS QUERY THEN　ＦＥＴＣＨ方式，在查询之前先分散计算各分片的词频信息，然后再执行查询，这种方式能提供更加准确的结果。
"""


def read_image(image_ids):
    df = pd.read_csv('reverse_image_search.csv')
    id_img = df.set_index('id')['path'].to_dict()
    imgs = []
    decode = ops.image_decode.cv2('rgb')
    for image_id in image_ids:
        path = id_img[image_id]
        imgs.append(decode(path))
    return imgs


search_pipeline = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='model', modality='text'))
    .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
    .map('vec', 'result', ops.ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='text_image_search', limit=5))
    .map('result', 'image_ids', lambda x: [item[0] for item in x])
    .output('image_ids')
)

def search(text):
    df = pd.read_csv('reverse_image_search.csv')
    id_img = df.set_index('id')['path'].to_dict()
    imgs = []
    image_ids = search_pipeline(text).to_list()[0][0]
    return [id_img[image_id] for image_id in image_ids]


def multi_recall(query_text):
    milvus_results,url = get_embedding(query_text)
    es_results = search_elasticsearch(query_text)
    function_calling_results = call_with_messages(query_text)
    all_results = milvus_results + es_results + function_calling_results
    all_results.sort(key=lambda x: x[1], reverse=True)
    response = Generation.call(
        api_key="",
        model='qwen-plus',  
        messages=messages,
        temperature=0.6,
    )
    return response

if __name__ == "__main__":

    query_text = ""
    
    results,url = multi_recall(query_text)

    messages = [
            {
                "content": "你是一个心思缜密的人，根据你所拥有的知识来做回答，对于不清楚或者不确定的问题，请用'不知道'来回答。",
                "role": "system"
            },
            {
                "role":"assistant",
                "content": "请根据以下信息进行回答：f{results},这是你查询到的信息中可能包含的URL：{url}，如果它不为空，请在你的回答的结尾把相关url拼接上去，如果为空直接忽略，不做拼接处理。"
            },
            {
                "role": "user",
                "content": f"{query_text}"
            }
    ]

    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Application")
        
        with gr.Row():
            input_text = gr.Textbox(label="Enter your query:", placeholder="Type your query here...")
        
        with gr.Row():
            output_text = gr.Textbox(label="Response:")
        
        submit_button = gr.Button("Submit Query")
        
        # 设置事件处理程序
        submit_button.click(fn=lambda x: multi_recall(x), inputs=input_text, outputs=output_text)
        interface = gr.Interface(search, 
                             gr.Textbox(lines=1),
                             [gr.Image(type="filepath", label=None) for _ in range(5)]
                            )

    interface.launch(inline=True, share=True)
    # 启动 Gradio 应用
    demo.launch()    



