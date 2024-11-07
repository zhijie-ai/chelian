# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/9/4 16:27
# @User   : RANCHODZU
# @File   : router_chain1.py
# @e-mail : zushoujie@ghgame.cn
# https://blog.csdn.net/sinat_29950703/article/details/136216665?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7Ebaidujs_utm_term%7ECtr-1-136216665-blog-133973011.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7Ebaidujs_utm_term%7ECtr-1-136216665-blog-133973011.235%5Ev43%5Epc_blog_bottom_relevance_base9&utm_relevant_index=1

from langchain.chains.llm import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RouterTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.llms.tongyi import Tongyi
import langchain
langchain.debug = True

# ollama_llm = Ollama(model='qwen:7b')
ollama_llm = Tongyi()

flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

prompt_infos = [
    {
        "key": "flower_care",
        "description": "适合回答关于鲜花护理的问题",
        "template": flower_care_template,
    },
    {
        "key": "flower_decoration",
        "description": "适合回答关于鲜花装饰的问题",
        "template": flower_deco_template,
    }
]

chain_map = {}
for info in prompt_infos:
    prompt = PromptTemplate(
        template=info['template'],
        input_variables=['input']
    )
    print("目标提示\n", prompt)

    chain = LLMChain(llm=ollama_llm, prompt=prompt, verbose=True)
    chain_map[info['key']] = chain

destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = RouterTemplate.format(destinations='\n'.join(destinations))
print("路由模板:\n", router_template)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=['input'],
    output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(ollama_llm, router_prompt, verbose=True)
# 构建默认链 default_chain
default_chain = ConversationChain(
    llm=ollama_llm,
    output_key="text",
    verbose=True
)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True
)

print(chain.run("如何为玫瑰浇水？"))
