#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/12 13:32                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    # train_set.append((reviewerID,hist,pos_list[i],1))
    # train_set.append((reviewerID,hist,neg_list[i],0))
    u, i, y, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2])#点击的
      y.append(t[3])#标签 1
      sl.append(len(t[1]))#历史的长度
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, y, hist_i, sl)

class DataInputTest:
    def __init__(self,data,batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0


    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i*self.batch_size:min((self.i+1) * self.batch_size,len(self.data))]
        self.i += 1

        u,i,j,sl = [],[],[],[]

        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, j, hist_i, sl)