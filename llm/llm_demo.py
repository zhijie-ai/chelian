# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/19 14:44
# @User   : RANCHODZU
# @File   : llm_demo.py
# @e-mail : zushoujie@ghgame.cn

import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-63a7f867881542df8c37b032cdb9963a'

from langchain_community.llms import Tongyi
from langchain_community.llms.baichuan import BaichuanLLM
from langchain_community.llms import QianfanLLMEndpoint
from langchain.agents import AgentType

llm = Tongyi()
# print(llm('怎么评价人工智能'))

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader('E:\\data\\llm/大模型技术白皮书2023版.pdf')
document = loader.load()
print(len(document))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_documents = text_splitter.split_documents(document)
print(len(split_documents))
chain = load_summarize_chain(llm, chain_type='stuff', verbose=True)
chain.run(split_documents)
print('*'*100)
# from langchain.vectorstores import Chroma
# from langchain.embeddings import LocalAIEmbeddings
# embedding = LocalAIEmbeddings()
# embs = Chroma.from_documents(split_documents, embedding)
# print(embs)
