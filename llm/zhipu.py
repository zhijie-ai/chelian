# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/2 18:20
# @User   : RANCHODZU
# @File   : zhipu.py
# @e-mail : zushoujie@ghgame.cn
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
os.environ['ZHIPUAI_API_KEY'] = '1a9c8e8902a0f59cd0959823a17ff44c.850suzsUIuXGL6Zr'

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)


def chat1():
    messages = [
        AIMessage(content="Hi."),
        SystemMessage(content="你的角色是一个诗人."),
        HumanMessage(content="用七言绝句的形式写一首关于AI的诗."),
    ]
    res = chat.invoke(messages)
    print(res)

def chat2():
    from langchain_core.callbacks.manager import CallbackManager
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.prompts import ChatPromptTemplate

    messages = [
        AIMessage(content="Hi."),
        SystemMessage(content="你的角色是一个诗人."),
        HumanMessage(content="用律诗的形式写一首关于AI的诗."),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    streaming_chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    streaming_chat(messages)

def chat3():
    from langchain_community.chat_models import ChatZhipuAI
    from langchain import hub
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    os.environ["TAVILY_API_KEY"] = "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"

    llm = ChatZhipuAI(temperature=0.01, model='glm-4')
    tools = [TavilySearchResults(max_results=2)]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": "黄龙江一派全都带蓝牙是什么意思?"})

def chat4():
    from zhipuai import ZhipuAI
    # from langchain_community.chat_models.zhipuai import ChatZhipuAI
    client = ZhipuAI(api_key='1a9c8e8902a0f59cd0959823a17ff44c.850suzsUIuXGL6Zr')

    response = client.chat.completions.create(
    model='glm-4',
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为智谱开放平台创作一个吸引人的slogan"},
        {"role": "assistant", "content": "当然，为了创作一个吸引人的slogan，请告诉我一些关于您产品的信息"},
        {"role": "user", "content": "智谱AI开放平台"},
        {"role": "assistant", "content": "智启未来，谱绘无限一智谱AI，让创新触手可及!"},
        {"role": "user", "content": "创造一个更精准、吸引人的slogan"}
    ])

    print(response.choices[0].message)


if __name__ == '__main__':
    # chat3()
    chat4()