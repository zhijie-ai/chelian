# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/19 17:26
# @User   : RANCHODZU
# @File   : llm_demo3.py
# @e-mail : zushoujie@ghgame.cn
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_community.chains.llm_requests import LLMRequestsChain
from langchain.chains.llm import LLMChain
# from langchain.chains import LLMRequestsChain, LLMChain

#  爬取网页并输出JSON数据
llm = Tongyi()
template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是新浪财经A股上市公司的公司简介。
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Extracted:"""
prompt = PromptTemplate(input_variables=['requests_result'], template=template)
chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"
}
response = chain(inputs)
print(response['output'])

