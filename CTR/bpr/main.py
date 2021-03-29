#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/29 14:54                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from preprocessor import Data_preprocessor
from BPR import BPR
import pandas as pd

__author__ = "Bo-Syun Cheng"
__email__ = "k12s35h813g@gmail.com"

if __name__ == "__main__":
    data = pd.read_csv('ratings_small.csv')
    dp = Data_preprocessor(data)
    processed_data = dp.preprocess()

    bpr = BPR(processed_data)
    bpr.fit()