# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/16 17:55
# @User   : RANCHODZU
# @File   : demo2.py
# @e-mail : zushoujie@ghgame.cn
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
os.environ["ZHIPUAI_API_KEY"] = '1a9c8e8902a0f59cd0959823a17ff44c.850suzsUIuXGL6Zr'  # 填入你自己的key


def demo1():
    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
    )

    messages = [
        AIMessage(content="Hi."),
        SystemMessage(content="你的角色是一个诗人."),
        HumanMessage(content="用七言绝句的形式写一首关于AI的诗."),
    ]

    response = chat.invoke(messages)
    response = chat(messages)
    print(response)

def demo2():
    from langchain_core.callbacks.manager import CallbackManager
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    messages = [
        AIMessage(content="Hi."),
        SystemMessage(content="你的角色是一个诗人."),
        HumanMessage(content="用七言绝句的形式写一首关于AI的诗."),
    ]

    streaming_chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    res = streaming_chat(messages)
    print('===================')
    print(res)


if __name__ == '__main__':
    demo2()