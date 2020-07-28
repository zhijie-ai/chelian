#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/26 11:34                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import os
path = os.getcwd() +'./file/'

def data_generator(batch_size=2):
    X=[]
    while True:
        for f in os.listdir('./file'):

            X.append(open(path+f).read().strip())
            if len(X)==batch_size:
                yield X
                X=[]

def data_generator2(batch_size=3):
    X=[]
    while True:
        for f in open('file/a.txt'):

            X.append(f.strip())
            if len(X)==batch_size:
                yield X
                X=[]

img_generator = data_generator2()
for i in range(10):
    print(next(img_generator),i)