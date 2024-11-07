# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/19 17:45
# @User   : RANCHODZU
# @File   : llm_demo4.py
# @e-mail : zushoujie@ghgame.cn
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from langchain.llms import Tongyi
from langchain import LLMMathChain, SerpAPIWrapper

# 自定义agent中所使用的工具
llm = Tongyi()
# 初始化搜索链和计算链
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# 创建一个功能列表，指明这个 agent 里面都有哪些可用工具，agent 执行过程可以看必知概念里的 Agent 那张图
tools = [
    Tool(name='Search',
         func=search.run,
         description="useful for when you need to answer questions about current events"),
    Tool(name='Calculator',
         func=llm_math_chain.run,
         description="useful for when you need to answer questions about math")
]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")