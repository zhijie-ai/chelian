# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/23 11:42
# @User   : RANCHODZU
# @File   : batch.py
# @e-mail : zushoujie@ghgame.cn
from openai import OpenAI
client = OpenAI()
batch = client.files.create( # file-PRZTufMHPHZq2wWHw8Jmwm
  file=open("batch.jsonl", "rb"),
  purpose="batch"
)
# print(batch)
print(client.files.list())