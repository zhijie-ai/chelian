# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/9/5 14:37
# @User   : RANCHODZU
# @File   : qianwen.py
# @e-mail : zushoujie@ghgame.cn
from openai import OpenAI
import os


def get_response():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen2-72b-instruct",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}],
        temperature=0.8,
        top_p=0.8
        )
    print(completion.model_dump_json())

if __name__ == '__main__':
    get_response()