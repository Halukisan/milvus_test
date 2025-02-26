from init import init
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from zhipuai import ZhipuAI
from create import ForOuthandleLine
import logging
from pymilvus import Collection, db, connections
import gradio as gr
import re


java_keyVector=[]
mysql_keyVector=[]
redis_keyVector=[]
web_keyVector=[]
# 设置日志级别
logging.basicConfig(level=logging.INFO)
def generate_text(self, prompt):
        try:
            response = self.zhipuai_client.chat.completions.create(
                model="glm-4-plus",  # 使用需要调用的模型编码
                messages=[
                    {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                    {"role": "user", "content": f"{prompt}"}
                ],
            )
            if response and response.choices and len(response.choices) > 0:
                message_content = response.choices[0].message.content.strip() if hasattr(response.choices[0].message, 'content') else ''
                return message_content or "No response from the model."
            else:
                logging.warning("Unexpected response structure from ZhipuAI API.")
                return "Error: Unable to generate text."
        except Exception as e:
            logging.error(f"An error occurred during the API request: {e}")
            return "Error: An unexpected error occurred."
 
    
def RAGquery(user_query,min_score):
    # 对用户输入的问题进行向量化
    api_key = ""
    # 初始化 ZhipuAI 客户端
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
                model="embedding-2",  # 替换为您想要使用的模型名称
                input=user_query  # 传递单个文本
            )

    # 获取生成的 embedding 值
    embedding_value = response.data[0].embedding
    # 走责任链确定分块
    coll_name = ForOuthandleLine(java_keyVector, mysql_keyVector, redis_keyVector, web_keyVector, embedding_value)
    # 查询
    conn = connections.connect(host="127.0.0.1", port=19530)
    db.using_database("sample_db")
    # 加载集合，不加载就不能插入数据
    collection = Collection(coll_name)
    collection.load()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = client.collection.search(
                data=[user_query],
                anns_field="embedding",
                param=search_params,
                limit=5,  # 返回前5个最相似的结果
                output_fields=["description"] 
            )

    print(results)
    # 整理搜索结果，并筛选出分数大于等于min_score的数据
    relevant_data = []
    for hits in results:
        print(f"hits:{hits}")
        for hit in hits:
            if hit.distance > min_score:
                data_info = {field: getattr(hit.entity, field) for field in ["description"]}
                relevant_data.append(data_info["description"])
                print(f"hitId:{hit.id},relevant_data:{relevant_data}")
            # 如果没有找到相关性足够高的数据，则不构建context
            if not relevant_data:
                print("没有找到相关性足够高的数据，不构建context！")
                final_answer = generate_text(user_query)
                return final_answer
            else:
                # 使用 ZhipuAI 的语言模型生成综合回答
                context = "\n".join(relevant_data)
                prompt = f"Based on the following context:\n{context}\nAnswer the question: {user_query}"
                final_answer = generate_text(prompt)
                return final_answer


    # 返回结果


def front():
   # 创建 Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Application")
        
        with gr.Row():
            input_text = gr.Textbox(label="Enter your query:", placeholder="Type your query here...")
        
        with gr.Row():
            output_text = gr.Textbox(label="Response:")
        
        submit_button = gr.Button("Submit Query")
        
        # 设置事件处理程序
        submit_button.click(fn=lambda x: RAGquery(x, min_score=0.6), inputs=input_text, outputs=output_text)
    # 启动 Gradio 应用
    demo.launch()
if __name__ == '__main__':
    front()
    