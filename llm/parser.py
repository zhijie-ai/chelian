# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/28 18:12
# @User   : RANCHODZU
# @File   : tmp.py
# @e-mail : zushoujie@ghgame.cn

# 注意PydanticOutputParser和StructuredOutputParser构造parser的区别
from typing import List
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOllama
from langchain.llms.ollama import Ollama
from langchain.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import ChatZhipuAI
from langchain.llms import Tongyi
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字")
    author_name: str = Field(description="书籍的作者")
    genres: List[str] = Field(description="书籍的体裁", type='array')


parser = PydanticOutputParser(pydantic_object=BookInfo)
print(parser.get_format_instructions())

response_schemas = [
    ResponseSchema(name="book_name", description="书籍的名字"),
    ResponseSchema(name="author_name", description="书籍的作者"),
    ResponseSchema(name="genres", description="书籍的体裁")
]
response_schemas = [
    {
        "name": "name",
        "type": "string",
        "description": "The name of the fruit"
    },
    {
        "name": "color",
        "type": "string",
        "description": "The color of the fruit"
    }
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 获取响应格式化的指令
format_instructions = output_parser.get_format_instructions()
print('===='*10)
print(format_instructions)
messages = [
    SystemMessage(content="你的角色是一个诗人."),
    AIMessage(content="Hi."),
    HumanMessage(content="用律诗的形式写一首关于AI的诗."),
]
model = ChatTongyi()
# model = ChatZhipuAI()
# model = Tongyi()
# print(model(ChatPromptTemplate.from_messages(messages).format()))  # 模型的输入是一个字符串
print(ChatPromptTemplate.from_messages(messages).format_messages())
print(model(ChatPromptTemplate.from_messages(messages).format_messages()))  # 模型的输入是一个message的list
