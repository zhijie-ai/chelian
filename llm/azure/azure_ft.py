# !/user/bin/python
# coding  : utf-8
# @Time   : 2025/1/7 14:56
# @User   : RANCHODZU
# @File   : azure_ft.py
# @e-mail : zushoujie@ghgame.cn
from openai import AzureOpenAI
import json
import time

url = f'https://azure-ran-resource-test001.openai.azure.com/'
key = f'9nFYPivGPXUwD9qZO88TPIjMTVh0evDYcoX0h033fEh0mXDos8LdJQQJ99BAACHYHv6XJ3w3AAABACOG2sJo'

# Load the training set
with open('demo.jsonl', 'r', encoding='utf-8') as f:
    training_dataset = [json.loads(line) for line in f]

# Training dataset stats
print("Number of examples in training set:", len(training_dataset))
print("First example in training set:")
for message in training_dataset[0]["messages"]:
    print(message)


client = AzureOpenAI(
  azure_endpoint=url,
  api_key=key,
  api_version="2024-08-01-preview"  # This API version or later is required to access seed/events/checkpoint features
)

# 上传数据
training_response = client.files.create(
    file=open('demo.jsonl', "rb"), purpose="fine-tune"
)
training_file_id = training_response.id
print("Training file ID:", training_file_id)

try:
    time.sleep(5)
except Exception as e:
    print(e)
# 开始微调
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-4o-2024-08-06", # Enter base model name. Note that in Azure OpenAI the model name contains dashes and cannot contain dot/period characters.
    seed=105 # seed parameter controls reproducibility of the fine-tuning job. If no seed is specified one will be generated automatically.
)

job_id = response.id
print("Job ID:", response.id)
print("Status:", response.status)
print(response.model_dump_json(indent=2))