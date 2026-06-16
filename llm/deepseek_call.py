#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author    : Admin
# file      : deepseek_call.py
# time      : 2026/5/8 16:07
# file_desc :
# e-mail : zushoujie@ghgame.cn

# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
print(os.environ.get('DEEPSEEK_API_KEY'))

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "光谷长江大桥现在有啥最新的动态"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)
