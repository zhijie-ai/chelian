# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/19 18:29
# @User   : RANCHODZU
# @File   : llm_demo5.py
# @e-mail : zushoujie@ghgame.cn
from langchain.memory import ChatMessageHistory
from langchain_community.chat_models import ChatZhipuAI

chat = ChatZhipuAI()

history = ChatMessageHistory()
history.add_ai_message('你好')
history.add_user_message('中国的首都在哪个城市')

ai_res = chat(history.messages)
print(ai_res)