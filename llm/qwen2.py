# !/user/bin/python
# coding  : utf-8
# @Time   : 2025/9/11 17:48
# @User   : RANCHODZU
# @File   : qwen2.py
# @e-mail : zushoujie@ghgame.cn
import os
import dashscope
messages = [
{
    "role": "system",
    "content": [
    {"text": "You are a helpful assistant."}]
},
{
    "role": "user",
    "content": [
    {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
    {"text": "图中描绘的是什么景象?"}]
}]
response = dashscope.MultiModalConversation.call(
    #若没有配置环境变量， 请用百炼API Key将下行替换为： api_key ="sk-xxx"
    api_key = os.getenv('DASHSCOPE_API_KEY'),
    model = 'qwen-vl-max-latest',  # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages = messages
)
print(response)