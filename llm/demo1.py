# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/16 16:50
# @User   : RANCHODZU
# @File   : demo1.py
# @e-mail : zushoujie@ghgame.cn
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Tongyi
import os
# os.environ['DASHSCOPE_API_KEY'] = 'sk-63a7f867881542df8c37b032cdb9963a'

# 定义响应的结构(JSON)，两个字段 answer和source。
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source referred to answer the user's question, should be a website.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 获取响应格式化的指令
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=['question'],
    partial_variables={"format_instructions": format_instructions}
)
model=Tongyi()
response = prompt.format_prompt(question="what's the capital of France?")
print('------------')
print(type(response), response)
output = model(response.to_string())
out = model.invoke(response)
print('======', out)
print('', response|model)
print(response.to_string())