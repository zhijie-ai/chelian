#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/20 17:25
 =================知行合一=============
 测试python装饰器
'''

import time

def deco(func):
    def wapper(*args,**kwargs):
        start = time.time()
        func(*args,**kwargs)
        end = time.time()
        print('time is %d s' %(end-start))
    return wapper

@deco
def func(a,b):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" % (a + b))


@deco
def func2(a,b,c):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" %(a+b+c))


if __name__ == '__main__':
    func2(3,4,5)
    func(3,4)

