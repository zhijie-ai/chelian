# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/19 16:09
# @User   : RANCHODZU
# @File   : upload.py
# @e-mail : zushoujie@ghgame.cn
from openai import OpenAI
client = OpenAI(api_key=f'', base_url=f'https://momi.qq.com/v1/files')
# client = OpenAI()
res = client.files.create(
  file=open("demo.jsonl", "rb"),
  purpose="fine-tune"
)
print(res)
print(client.files.list())
# print(client.files.retrieve('file-Vm8tNkHXNsfep9h84xaVa6'))
print(client.files.content('file-Vm8tNkHXNsfep9h84xaVa6'))
# client.fine_tuning.jobs.create(training_file=f'file-Vm8tNkHXNsfep9h84xaVa6', model=f'gpt-4o-mini-2024-07-18')
