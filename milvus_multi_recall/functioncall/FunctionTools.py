import os
from dashscope import Generation
from langchain_community.utilities import SerpAPIWrapper
from datetime import datetime
import random
import json
import requests


# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                # 查询天气时需要提供位置，因此参数设置为location
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                }
            },
            "required": [
                "location"
            ]
        }
    },
    # 工具3 Google搜索
    {
        "type": "function",
        "function": {
            "name": "google_search_action",
            "description": "在Google搜索一些信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户想要搜索的问题"
                    }
                }
            },
            "required": ["query"]
        } 
    }
]

def google_search_action(query: str):
    search = SerpAPIWrapper()
    result = search.run(query)

    return result
# 模拟天气查询工具。返回结果示例：“北京今天是晴天。”
def get_current_weather(location):
    apiUrl = ''
    apiKey = ''
    request = {
        'key': apiKey,
        'city': location,
    }
    response = requests.get(apiUrl, params=request)
    if response.status_code == 200:
        return response.json()
    else:
        return f"查询{location}天气失败，请稍后再试。"


# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"


# 封装模型响应函数
def get_response(messages):
    response = Generation.call(
        api_key="",
        model='qwen-plus',  
        messages=messages,
        tools=tools,
        parallel_tool_calls=True,
        temperature=0.6,
    )
    return response


def call_with_messages(message):
    print('\n')
    messages = [
            {
                "content": "你是一个心思缜密的人，根据你所拥有的知识来做回答，对于不清楚或者不确定的问题，请用'不知道'来回答。",
                "role": "system"
            },
            {
                "role": "user",
                "content": f"{message}"
            }
    ]
   
    # 模型的第一轮调用
    first_response = get_response(messages)
    assistant_output = first_response.output.choices[0].message
    messages.append(assistant_output)
    if 'tool_calls' not in assistant_output:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        return
    # 如果模型选择的工具是get_current_weather
    elif assistant_output.tool_calls[0]['function']['name'] == 'get_current_weather':
        tool_info = {"name": "get_current_weather", "role":"tool"}
        location = json.loads(assistant_output.tool_calls[0]['function']['arguments'])['location']
        tool_info['content'] = get_current_weather(location)
    # 如果模型选择的工具是get_current_time
    elif assistant_output.tool_calls[0]['function']['name'] == 'get_current_time':
        tool_info = {"name": "get_current_time", "role":"tool"}
        tool_info['content'] = get_current_time()
    elif assistant_output.tool_calls[0]['function']['name'] == 'google_search_action':
        tool_info = {"name": "google_search_action", "role":"tool"}
        query = json.loads(assistant_output.tool_calls[0]['function']['arguments'])['query']
        tool_info['content'] = google_search_action(query)
    print(f"工具输出信息：{tool_info['content']}\n")
    messages.append(tool_info)
    return messages

