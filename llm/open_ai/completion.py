# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/25 15:00
# @User   : RANCHODZU
# @File   : completion.py
# @e-mail : zushoujie@ghgame.cn
import openai, os
from openai import OpenAI
# client = OpenAI()
client = openai.Api()
# 似乎completion api已经废弃了
model = "text-ada-001"

response = client.completions.create(
  model=model,        # 模型类型，例如 "gpt-3.5-turbo" 或 "gpt-4"
  prompt="你好，世界！",  # 输入的提示文本
  max_tokens=100,       # 最多生成的 token 数量
  temperature=0.7,      # 随机性控制，值越高越随机
)

# 输出生成的文本
print(response.choices[0].text.strip())