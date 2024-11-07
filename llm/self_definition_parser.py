# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/28 16:08
# @User   : RANCHODZU
# @File   : self_definition_parser.py
# @e-mail : zushoujie@ghgame.cn

# 正常情况下llm的输入是字符串，chatlm的输入是message，这里展示了如何将message转为llm的输入
# message有2中情况，一种是[()],一种是通过AIMessage的形式
from typing import List
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import ChatPromptValue

from langchain.output_parsers import PydanticOutputParser, ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.llms import Tongyi


class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字")
    author_name: str = Field(description="书籍的作者")
    genres: List[str] = Field(description="书籍的体裁")


parser = PydanticOutputParser(pydantic_object=BookInfo)
# 查看输出解析器的内容，会被输出成json格式
print(parser.get_format_instructions())
print('='*100)

response_schemas = [
    ResponseSchema(name="book_name", description="书籍的名字"),
    ResponseSchema(name="author_name", description="书籍的作者"),
    ResponseSchema(name="genres", description="书籍的体裁", type='array')
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions} 你输出的结果请使用中文。"),
    ("human", "请你帮我从书籍的概述中，提取书名、作者，以及书籍的体裁。书籍概述会被三个#符号包围。\n###{book_introduction}###")
])

book_introduction = """
《朝花夕拾》原名《旧事重提》，是现代文学家鲁迅的散文集，收录鲁迅于1926年创作的10篇回忆性散文， [1]1928年由北京未名社出版，现编入《鲁迅全集》第2卷。
此文集作为“回忆的记事”，多侧面地反映了作者鲁迅青少年时期的生活，形象地反映了他的性格和志趣的形成经过。前七篇反映他童年时代在绍兴的家庭和私塾中的生活情景，后三篇叙述他从家乡到南京，又到日本留学，然后回国教书的经历；揭露了半殖民地半封建社会种种丑恶的不合理现象，同时反映了有抱负的青年知识分子在旧中国茫茫黑夜中，不畏艰险，寻找光明的困难历程，以及抒发了作者对往日亲友、师长的怀念之情 [2]。
文集以记事为主，饱含着浓烈的抒情气息，往往又夹以议论，做到了抒情、叙事和议论融为一体，优美和谐，朴实感人。作品富有诗情画意，又不时穿插着幽默和讽喻；形象生动，格调明朗，有强烈的感染力。
"""

model = Tongyi()
final_prompt = prompt.invoke({"book_introduction": book_introduction,
                              "parser_instructions": output_parser.get_format_instructions()})
print(type(final_prompt), type(prompt))
response = model.invoke(final_prompt)
print('*'*100)
print(response, type(response))
print('*'*100)
result = output_parser.invoke(response)
print(result, type(result))  # <class 'dict'>
result = parser.invoke(response)  # <class '__main__.BookInfo'>
print(result, type(result))
print('-'*100)
# p = prompt.format(book_introduction=book_introduction,
#                   parser_instructions=output_parser.get_format_instructions())
# print(model(p))
