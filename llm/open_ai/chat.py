# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/25 15:13
# @User   : RANCHODZU
# @File   : chat.py
# @e-mail : zushoujie@ghgame.cn
from openai import OpenAI
import os
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)

# completion = client.completions.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt="<prompt>"
# )
# print(completion)
