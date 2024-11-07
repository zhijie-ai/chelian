# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/23 14:01
# @User   : RANCHODZU
# @File   : tool_definition3.py
# @e-mail : zushoujie@ghgame.cn
import langchain
from langchain.chat_models.tongyi import ChatTongyi
from langchain.agents import initialize_agent, tool, Tool, load_tools, AgentType

langchain.debug = True

llm = ChatTongyi()

tools = load_tools(['llm-math', 'wikipedia'], llm=llm)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    # 设置agent类型，CHAT表示agent使用了chat大模型，REACT表示在prompt生成时使用更合乎逻辑的方式，获取更严谨的结果
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # 设置agent可以自动处理解析失败的情况，由于使用大模型处理逻辑，返回值并不会100%严谨，所以可能会出现返回的额数据不符合解析的格式，导致解析失败
    # agent可以自动处理解析失败的情况，再次发送请求以期望获取到能正常解析的返回数据
    handle_parsing_erros=True,
    verbose=True
)

res = agent("1000的35%是多少？")
print(res)
