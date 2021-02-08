#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/10 16:51                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

class T():
    def __init__(self,name,age):
        self.name=name
        self.age = age

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __repr__(self):
        return "T:{},name:{},age:{}".format(hash(self), self.name,
                                                              self.age)
    def __cmp__(self, other):
        pass

    def __le__(self, other):
        pass

    def __int__(self):
        pass
imgs = range(10000)
import numpy as np
def data_generator(batch_size = 32):
    X=[]
    while True:
        for f in imgs:
            X.append(f)
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X=[]

class img_generator:

    def __len__(self):
        return self.steps

    def __iter__(self,batch_size=64):
        X = []
        while True:
            for f in imgs:
                X.append(f)
                if len(X) == batch_size:
                    X = np.array(X)
                    yield X
                    X = []

if __name__ == '__main__':
    # t1 = T('a',20)
    # t2 = T('a',20)
    # t3 = T('b',10)
    # t4 = T('b',20)
    # l = [t1,t3,t4]
    # print(t2 in l)
    # print(T.__dict__)
    # print(t1.__dict__)
    # print(T.__module__)
    # print(t1.__module__)
    # print(t1.__class__)

    # for i in range(5):
    #     print(i)
    # else:
    #     print('aaaaaaaa')

    # for i in data_generator():
    #     print(i)
    # d = data_generator()
    # print(next(d))
    # print(next(d))
    gen=img_generator().__iter__()
    for i in range(5):
        print(next(gen))