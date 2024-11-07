# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/29 11:28
# @User   : RANCHODZU
# @File   : demo1.py
# @e-mail : zushoujie@ghgame.cn

from langchain.chat_models.tongyi import ChatTongyi
from langchain.llms import Tongyi
from langchain_core.messages import AIMessage, AIMessageChunk

model = ChatTongyi()
m = Tongyi()
def parse(ai_message: AIMessage) -> str:
    return ai_message.content.swapcase()

chain = model|parse
res = model.invoke('hello')
r = m.invoke('hello')
print(type(r), r)
print(type(res), res)
print(chain.invoke('hello'))