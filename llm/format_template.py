# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/28 17:45
# @User   : RANCHODZU
# @File   : format_template.py
# @e-mail : zushoujie@ghgame.cn

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
messages = [
    SystemMessage(content="你的角色是一个诗人."),
    AIMessage(content="Hi."),
    HumanMessage(content="用律诗的形式写一首关于AI的诗."),
]
mess = ChatPromptTemplate.from_messages(messages).invoke({})
print(mess, type(mess))
print('*'*100)
messages = [
    SystemMessage(content="你的角色是一个诗人."),
    AIMessage(content="Hi."),
    HumanMessage(content="用律诗的形式写一首关于{AI}的诗."),
]
template = ChatPromptTemplate.from_messages(messages)
print(template, type(template))
print('*'*100)
m1 = template.format(AI='AI')
m2 = template.format_messages(AI='ai')
print(type(m1), m1)
print(type(m2), m2)
print('*'*100)

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format(name="Bob", user_input="What is your name?")
# messages = chat_template.invoke({'name': 'Bob', 'user_input': "What is your name?"})
m2 = chat_template.invoke({'name': 'Bob', 'user_input': "What is your name?"})
print(messages, type(messages))  # str
print(m2, type(m2))  # ChatPromptValue
print('*'*100)

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, \
    HumanMessagePromptTemplate

c1 = SystemMessagePromptTemplate.from_template("You are a helpful AI bot. Your name is {name}.")
c2 = HumanMessagePromptTemplate.from_template("You are a helpful AI bot. Your name is {name}.")
c3 = AIMessagePromptTemplate.from_template("You are a helpful AI bot. Your name is {name}.")
c4 = HumanMessagePromptTemplate.from_template("You are a helpful AI bot. Your name is {name}.")
chat_template = ChatPromptTemplate.from_messages([c1, c2, c3, c4])
# 效果如上
print('AAAAAAAAAAAA')
messages = chat_template.format(name="Bob", user_input="What is your name?")
print(type(messages), messages)  # str,
messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
print(type(messages), messages)  # list
