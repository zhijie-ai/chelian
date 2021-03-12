#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/11 9:20                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import os
'''
将libsvm格式的数据重新整理成一个dict，key为libsvm格式中的特征(name:1) value为新的index,此时，这些数据就代表原始
    数据中onehot之后值为1的特征
'''

class LoadData(object):
    '''given the path of data, return the data format for AFM and FM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type="square_loss"):
        self.path = path + dataset + "/"

        self.trainfile = self.path + dataset +".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".valid.libfm"

        self.features_M = self.map_features( )#将ligsvm格式的数据整理成一个dict
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        # print("features_M:", len(self.features))
        return  len(self.features)

    def read_features(self, file): # read a feature file -1.0 84982:1 58:1 39525:1
        f = open( file )
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.features:
                    self.features[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Train:" , len(Y_))

        X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)#X为index(并非libsvm中的index，而是重新生成的index)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Validation:", len(Y_))

        X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Test:", len(Y_))

        return Train_data,  Validation_data,  Test_data

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open( file )
        X_ = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )#label 为1和-1

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )
            line = f.readline()
        f.close()
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [ Y_[i] for i in indexs]
        Data_Dic['X'] = [ X_[i] for i in indexs]
        return Data_Dic

    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in range(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in range(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in range(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in range(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable

if __name__ == '__main__':
    path = '../data/'
    dataset ='frappe'
    ld = LoadData(path,dataset)
    print(ld.features_M)
    f1 = path + dataset +"/frappe.train.libfm"
    f2 = path + dataset + "/frappe.test.libfm"
    f3 = path + dataset + "/frappe.valid.libfm"
    max1 = max(set([int(item.split(':')[0])  for line in open(f1,'r') for item in line.strip().split(' ')[1:]]))
    max2 = max(set([int(item.split(':')[0])  for line in open(f2,'r') for item in line.strip().split(' ')[1:]]))
    max3 = max(set([int(item.split(':')[0])  for line in open(f3,'r') for item in line.strip().split(' ')[1:]]))
    # print(max(max1,max2,max3))
    print(max1)