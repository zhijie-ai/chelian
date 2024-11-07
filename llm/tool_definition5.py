# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/23 14:01
# @User   : RANCHODZU
# @File   : tool_definition3.py
# @e-mail : zushoujie@ghgame.cn
import langchain
from langchain.chat_models.tongyi import ChatTongyi
from langchain.llms import Tongyi
from langchain.agents import initialize_agent, tool, Tool
from pypinyin import pinyin

langchain.debug = True

llm = ChatTongyi()
llm = Tongyi()


@tool
def chinese_to_pinyin(param: str) -> str:
    # 方法名作为自定义tool的实例名称
    # query参数是经过大模型分析之后，送入当前tool的文本信息
    # 方法中必须要存在doc,这个doc会被作为tool的描述信息，提交给大模型用于判断什么时候怎么调用当前tool
    """接收中文文本，返回对应中文的拼音列表，能够将中文转换成拼音的工具，必须要接收一个中文文本作为输入参数，并且返回的时候总是一个列表数据"""
    r = pinyin(param)
    # 将转换结果的格式修正一下[["zhong"],["wen"]] => "['zhong','wen']"
    return str([i[0] for i in r])


agent = initialize_agent(
    tools=[chinese_to_pinyin],  # 设置可以使用的工具列表
    llm=llm, vervose=True
)


# 待翻译文本
chinese_str = "打开Debug模式，将会输出更详细的日志信息，方便了解整体运行逻辑"
command_str = f"将以下文本转换成拼音: \n{chinese_str}"
res = agent.run(command_str)
print(res)
