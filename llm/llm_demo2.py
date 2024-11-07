# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/19 17:04
# @User   : RANCHODZU
# @File   : llm_demo2.py
# @e-mail : zushoujie@ghgame.cn
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import Tongyi

# 结构化输出
llm = Tongyi()
response_schemas = [ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
                    ResponseSchema(name="good_string", description="This is your response, a reformatted response")]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
print(output_parser.get_format_instructions())
template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""
prompt = PromptTemplate(input_variables=['user_input'],
                        partial_variables={'format_instructions': output_parser.get_format_instructions()},
                        template=template)
prom = prompt.format(user_input='welcom to califonya!')
llm_out = llm(prom)
print('*'*100)
print(llm_out)
print('*'*100)
print(output_parser.parse(llm_out))
print('====='*10)
prompt = prompt.format_prompt(user_input='welcom to califonya!')
llm_out = llm(prompt.to_string())
print(llm_out)
print('*'*100)
print(output_parser.parse(llm_out))

