import json
from pymilvus import Collection, db, connections
import numpy as np
from zhipuai import ZhipuAI
from pymilvus import MilvusClient
from dwspark.models import ChatModel
from typing import List
import gradio as gr # 通过as指定gradio库的别名为gr

SPARKAI_APP_ID = 'ce51a015'
SPARKAI_API_SECRET = 'ZGI3YjBhNDlhYzk0MmVmYjNmZjBkY2I0'
SPARKAI_API_KEY = '10d0425e02925168597cb69c1f437709'
# 加载sdk配置
from sparkai.core.messages import ChatMessage, AIMessageChunk
from dwspark.config import Config
config = Config(SPARKAI_APP_ID, SPARKAI_API_KEY, SPARKAI_API_SECRET)

# 替换为您的实际 API 密钥
api_key = "8fb21e517a965b10cf87b7fdadf18a74.UVLbxTJXdL3YLxcT"

# 小明发现了大型语言模型（LLM）的文本摘要功能，这个功能对小明来说如什么一样？
# 这个功能对小明来说如同灯塔一样
# 自定义函数，功能是随机选返回指定语句，并与用户输入的 chat_query 一起组织为聊天记录的格式返回
def chat(chat_query, chat_history):
        # 在How are you 等语句里随机挑一个返回，放到 bot_message 变量里
        # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        model = ChatModel(config, stream=False)
        # ans = model.generate([ChatMessage(
        #      role="user", content=chat_query
        #      )])
        # 定义系统角色的消息
        callBack = get_embedding(chat_query)
        system_message = ChatMessage(
             role="system", 
             content=f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
 ​
                 Question: {chat_query} 
                 ​
                 Context: "{callBack}" 
                 ​
                 Answer:
                 """
             )

        # 用户的问题
        user_message = ChatMessage(role="user", content=chat_query)

        # 构建消息列表
        messages: List[ChatMessage] = [system_message, user_message]

        # 调用模型生成回复
        ans = model.generate(messages)
        bot_message = ans
        # 添加到 chat_history 变量里
        chat_history.append((chat_query, bot_message))
        # 返回 空字符，chat_history 变量，空字符用于清空 chat_query 组件，chat_history 用于更新 chatbot组件
        return "", chat_history
def get_embedding(question):
    # 初始化 ZhipuAI 客户端
    client = ZhipuAI(api_key=api_key)
    # 这里text的内容即为我们需要从向量数据库中检索的内容，通常为用户的问题
    text  = [question]
    conn = connections.connect(host="127.0.0.1", port=19530)
    db.using_database("sample_db")
    coll_name = 'word_vector'
    # 加载集合，不加载就不能插入数据
    collection = Collection(coll_name)
    collection.load()
    mids, embedings, counts, descs = [], [], [], []
    idx = 1
    # 调用嵌入创建 API
    response = client.embeddings.create(
        model="embedding-2",  # 替换为您想要使用的模型名称
        input=text  # 传递单个文本
    )
    # print([response.data[0].embedding])
    collection = Collection(coll_name)
    # 1. 设置一个Milvus客户端
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    # 确保集合已经被加载并且不是空的
    print(collection.is_empty)
    # 如果没有执行查询，检查一下collection是否存在，如果存在还不能查询，就重启一下
    if not collection.is_empty:
        # 构建查询参数
        print("开始查询")
        res = collection.search(
            anns_field="embeding",
            # 搜索的字段，这里假设有一个文本字段叫做 description
            data=[response.data[0].embedding],
            # 返回 TopK 个最相关的结果
            limit=3,
            param={"metric_type": "IP", "params": {}}, # 搜索参数
            # 下面的desc非常重要，如果没有下面这一句，则不会输出desc的内容。
            output_fields=["desc"] # 返回的输出字段
        )
        print(f"检索得到的答案为：{res}")
        return res
    return None

# gr.Blocks()：布局组件，创建并给了他一个名字叫 demo
with gr.Blocks() as demo:
    # gr.Chatbot()：输入输出组件，用于展示对话效果
    chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天历史")
    # gr.Textbox()：输入输出组件，用于展示文字
    chat_query = gr.Textbox(label="输入问题", placeholder="输入需要咨询的问题")
    # gr.Button：控制组件，用于点击，可绑定不同的函数触发处理
    llm_submit_tab = gr.Button("发送", visible=True)
    
    # gr.Examples(): 输入输出组件，用于展示组件的样例，点击即可将内容输入给 chat_query 组件
    gr.Examples(["请介绍一下Datawhale。", "如何在大模型应用比赛中突围并获奖？", "请介绍一下基于Gradio的应用开发"], chat_query)

    # 定义gr.Textbox()文字组件 chat_query 的 submit 动作(回车提交)效果，执行函数为 chat, 第一个[chat_query, chatbot]是输入，第二个 [chat_query, chatbot] 是输出
    chat_query.submit(fn=chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])
    # 定义gr.Button()控制组件 llm_submit_tab 的 点击动作 效果，执行函数为 chat, 第一个[chat_query, chatbot]是输入，第二个 [chat_query, chatbot] 是输出，效果与上一行代码同
    llm_submit_tab.click(fn=chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])

# 运行demo
if __name__ == '__main__':
    demo.queue().launch()


