# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/8/22 16:31
# @User   : RANCHODZU
# @File   : tool_definition.py
# @e-mail : zushoujie@ghgame.cn
"""
有4种方式定义tool
1. 使用@tool装饰器来修饰函数将其变成tool
2. 继承BaseTool自定义工具，通过继承BaseModel来定义参数
3. 通过StructuredTool.from_function定义工具
4. 通过Tool.from_function定义工具
5. tools = [
    Tool(name='Search',
         func=search.run,
         description="useful for when you need to answer questions about current events"),
    Tool(name='Calculator',
         func=llm_math_chain.run,
         description="useful for when you need to answer questions about math")
"""


from langchain.tools import BaseTool, StructuredTool, tool, Tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun


# 使用包装器自定义tool
@tool
def search(query: str) -> str:
    """Look up things online."""
    return 'LangChain'

print(search.name)
print(search.description)
print(search.args)


class SearchInput(BaseModel):
    param: str = Field(description='一个合法的查询参数')


@tool('search-tool', args_schema=SearchInput, return_direct=True)
def search(query: str) -> str:
    """Look up things online1."""
    return 'LangChain'


print(search.name)
print(search.description)
print(search.args)
print(search.return_direct)
print('*'*100)


class CalculatorInput(BaseModel):
    a: int = Field(description='第一个参数')
    b: int = Field(description='第二个参数')


"""使用类的方式自定义tool"""
class CustomCalculaterTool(BaseTool):
    name = 'Calculator'
    description = '面对数学计算式非常有用的'
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct = True

    def _run(self, a1: int, b1: int,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> int:
        """使用工具"""
        return a1 * b1

    async def _arun(self, a: int, b: int,
                    run_manager:Optional[AsyncCallbackManagerForToolRun] = None) -> NotImplementedError:
        """异步方式使用工具"""
        return NotImplementedError("不支持异步调用方式")


# 使用StructuredTool构造tool
def search_function(query: str):
    return 'LangChain'


if __name__ == '__main__':
    calculator = CustomCalculaterTool()
    print(calculator.name)
    print(calculator.description)
    print(calculator.args)
    print('*'*100)
    search = StructuredTool.from_function(
        func=search_function,
        name='Search',
        description='搜索api'
    )
    print(search.name)
    print(search.description)
    print(search.args)
    print('*'*100)
    search = Tool.from_function(
        func=search_function,
        name='Search1',
        description='搜索api'
    )
    print(search.name)
    print(search.description)
    print(search.args)
