# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/2 17:18
# @User   : RANCHODZU
# @File   : qianwen.py
# @e-mail : zushoujie@ghgame.cn

import os
os.environ['DASHSCOPE_API_KEY'] = ''
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

chatLLM = ChatTongyi(streaming=True)

def chat1():
    res = chatLLM.stream([HumanMessage(content="你好")], streaming=True)
    for r in res:
        print('chat resp:', r)

def chat2():
    messages = [
        SystemMessage(
            content=
            "You are a helpful assistant that translates English to Chinese."),
        HumanMessage(
            content=
            "Translate this sentence from English to Chinese. I love programming."
        ),
    ]
    res = chatLLM(messages)
    print(res)

def chat3():
    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        return first_int * second_int

    llm = ChatTongyi(model="qwen-turbo")
    llm_with_tools = llm.bind_tools([multiply])
    msg = llm_with_tools.invoke("5乘以32的结果是多少？").tool_calls
    print(msg)

def chat4():
    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    messages


if __name__ == '__main__':
    # chat1()
    chat3()