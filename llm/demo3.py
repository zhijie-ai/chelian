# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/9/5 16:25
# @User   : RANCHODZU
# @File   : demo3.py
# @e-mail : zushoujie@ghgame.cn
from langchain.llms import Ollama
from langchain_community.llms.ollama import Ollama
ollama = Ollama(base_url='http://localhost:8888',
model="llama2:latest")  # llama2:latest可以成功, llama2也可以成功
print(ollama("why is the sky blue"))