# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/7/24 18:16
# @User   : RANCHODZU
# @File   : qianfan_demo.py
# @e-mail : zushoujie@ghgame.cn
import os
import qianfan

payload = {"role":"user","content":"帮我生成一首唐诗。要求\n1. 以春天为主题\n2. 感情丰富，辞藻华丽"}

resp = qianfan.ChatCompletion().do(endpoint="completions", messages=[payload], temperature=0.95, top_p=0.8, penalty_score=1, disable_search=False, enable_citation=False, response_format="text")
print(resp.body)