# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/15 13:40
# @User   : RANCHODZU
# @File   : demo.py
# @e-mail : zushoujie@ghgame.cn

from langchain.chains import LLMChain
from langchain_community.chains import openapi
from langchain_openai import OpenAI
from langchain.llms import OpenAI
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.document_loaders import CSVLoader, UnstructuredCSVLoader
from langchain.document_loaders import PyPDFLoader
from pprint import pprint
state_of_the_union = """
斗之力，三段！”

    望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

    “萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

    中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

    “三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

    “哎，这废物真是把家族的脸都给丢光了。”

    “要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

    “唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator='\n\n', chunk_size=128, chunk_overlap=10, length_function=len)
texts = text_splitter.create_documents([state_of_the_union])
# pprint(texts)

metadatas = [{"document": 1}, {"document": 2}]
documents = text_splitter.create_documents([state_of_the_union, state_of_the_union], metadatas=metadatas)
pprint('========='*10)
pprint(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
pprint([e.value for e in Language])
pprint(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                        chunk_size=50, chunk_overlap=0)
python_docs = python_splitter.create_documents([PYTHON_CODE])
# print(python_docs)
pprint('*'*100)
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_overlap=10, chunk_size=128, length_function=len)
texts = text_splitter.create_documents([state_of_the_union])
# pprint(texts)
pprint('*'*100)
text_splitter = CharacterTextSplitter(chunk_overlap=10, chunk_size=128)
texts = text_splitter.split_text(state_of_the_union)
pprint(texts)
for text in texts:
    pprint(len(text))
pprint('*'*100)
from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
template = """/
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""
prompt = PromptTemplate.from_template(template)
r = prompt.format(product='colorful socks')
print(type(prompt),prompt)
print(type(r),r)
pprint('---*'*100)
import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-63a7f867881542df8c37b032cdb9963a'
from langchain.llms import Tongyi
# llm = Tongyi()
# res = llm.generate(["讲一个笑话", "朗读一首唐诗"]*15)
# print(res)
# pprint('*'*100)
from langchain.chains.llm import LLMChain
llm = Tongyi()
prompt = PromptTemplate(input_variables=['product'],
                        template='针对{product}这种产品，公司应该叫什么名字才更合理')
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run('花式足球'))
prompt = PromptTemplate(input_variables=['company','product'],
                        template="What is a good name for {company} that makes {product}?")
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({'company':'ABC Startup', 'product':'colorful socks'}))
pprint('*'*100)
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class Joke(BaseModel):
    setup: str = Field(description='question to set up a joke')
    punchline: str = Field(description="answer to resolve the joke")


parser = PydanticOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template='Answer the user query.\n{format_instructions}\n{query}\n',
    input_variables=['query'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
joke_query = 'Tell me a joke'
formatted_prompt = prompt.format_prompt(query=joke_query)
pprint(formatted_prompt.to_string())
pprint(parser.get_format_instructions())
pprint('*'*100)
output = llm(formatted_prompt.to_string())
pprint(output)
pprint(parser.parse(output))
