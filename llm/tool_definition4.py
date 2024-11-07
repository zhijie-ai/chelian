# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/23 15:13
# @User   : RANCHODZU
# @File   : tool_definition4.py
# @e-mail : zushoujie@ghgame.cn
import langchain
from langchain.chat_models.tongyi import ChatTongyi
from langchain.agents import initialize_agent
from langchain_experimental.tools.python.tool import PythonREPLTool

langchain.debug = True

llm = ChatTongyi(temperature=0)
agent = initialize_agent(
    tools=[PythonREPLTool()],
    llm=llm, verbose=True
)

# name_list = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez",
#              "Łukasz Kaiser", "Illia Polosukhin"]
name_list = ["Ashish Vaswani", "Noam Shazeer"]
command_str = f"将下列人名优先用姓氏、再使用名字进行排序，并将结果打印出来: \n{name_list}"
res = agent.run(command_str)
print(res)