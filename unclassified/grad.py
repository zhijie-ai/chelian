#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/3 10:18
 =================知行合一=============
'''

a=0.001             #定义收敛步长
xd=1                #定义寻找步长
x=0                 #定义一个种子x0
i=0                 #循环迭代次数
y=0
dic={}
import math
def f(x):
    y=math.sin(x)   #定义函数f(X)=sinx
    return y

def fd(x):
    y=math.cos(x)   #函数f(x)导数fd(X)=cosx
    return y

while y>=0 and y<3.14*4:
    y=y+xd
    x=y
    while abs(fd(x))>0.001: #定义精度为0.001
        x=x+a*fd(x)
    if x>=0 and x<3.14*4:
        print(x,f(x))
        dic[y]=x
print(dic)
ls=[]
for i in dic.keys():
    cor=0
    if ls is None:
        ls.append(dic[i])
    else:
        for j in ls:
            if dic[i]-j<0.1:
                cor=1
                break
        if cor==0:
            ls.append(dic[i])
print(ls)