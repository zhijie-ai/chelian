# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/22 17:25
# @User   : RANCHODZU
# @File   : tool_definition2.py
# @e-mail : zushoujie@ghgame.cn
import langchain
langchain.debug = True

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import Tongyi

def buy_xlb(days: int):
    return '成功'


def buy_jz(input: str):
    return '成功'


xlb = Tool.from_function(func=buy_xlb,
                         name="buy_xlb",
                         description="当你需要买一份小笼包时候可以使用这个工具,他的输入为帮我买一份小笼包,他的返回值为是否成功"
                         )
jz = Tool.from_function(func=buy_jz,
                        name="buy_jz",
                        description="当你需要买一份饺子时候可以使用这个工具,他的输入为帮我买一份饺子,他的返回值为是否成功"
                        )
tools = [xlb, jz]

llm = Tongyi(temprature=0)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print(agent.run('帮我买一份饺子'))
