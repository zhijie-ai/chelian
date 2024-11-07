# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/19 18:51
# @User   : RANCHODZU
# @File   : llm_demo6.py
# @e-mail : zushoujie@ghgame.cn
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

db = SQLDatabase.from_uri("sqlite:///db/user.db")
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("Describe the playlisttrack table")
