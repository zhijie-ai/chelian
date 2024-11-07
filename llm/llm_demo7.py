# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/20 15:59
# @User   : RANCHODZU
# @File   : llm_demo7.py
# @e-mail : zushoujie@ghgame.cn
import langchain
langchain.debug = True

from langchain import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms import Tongyi

# db = SQLDatabase.from_uri("mysql+pymysql://llm:llm@192.168.11.66/llm")
db = SQLDatabase.from_uri("mysql+pymysql://llm:llm@192.168.11.137:3306/mysql")
llm = Tongyi(temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# agent_executor = create_sql_agent(
#     llm=Tongyi(temperature=0),
#     toolkit=toolkit,
#     verbose=True
# )
agent_executor = create_sql_agent(
    llm=Tongyi(temperature=0),
    db=db,
    verbose=True
)
print(agent_executor.run("当前数据库总共有多少条数据"))