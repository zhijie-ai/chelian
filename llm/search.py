#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author    : Admin
# file      : search.py
# time      : 2026/5/27 15:37
# file_desc :
# e-mail : zushoujie@ghgame.cn
import requests

# 请替换为你在百度智能云控制台获取的真实API Key
API_KEY = "bce-v3/ALTAK-MqKFM8SMQH7ZFwsMTJmcp/b656b3cd128ef32201ed89a9f2d218867e506b05"
url = "https://qianfan.baidubce.com/v2/ai_search/web_search"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "messages": [
        {"role": "user", "content": "光谷长江大桥的最新进展"}
    ],
    "search_source": "baidu_search_v2", # 使用百度搜索
    "resource_type_filter": [{"type": "web", "top_k": 10}], # 返回10条网页结果
    # "search_recency_filter": "month" # 可选：筛选最近一月的结果
}

response = requests.post(url, headers=headers, json=payload)
data = response.json()

# 提取搜索结果
if 'references' in data:
    for item in data['references']:
        print(f"标题: {item['title']}")
        print(f"链接: {item['url']}")
        print(f"摘要: {item['content']}")
        print("-" * 20)