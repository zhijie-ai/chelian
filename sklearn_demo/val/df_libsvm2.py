#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/8/11 8:57                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
from sklearn.datasets import dump_svmlight_file

df = pd.DataFrame({'userid':[50,5,6,7,14,20,22,23,34,36],
                   'state':[1,1,2,2,1,2,1,1,2,4],
                   'gender':[1,1,1,1,2,1,2,1,1,3],
                   'age':[20,30,10,50,60,10,20,80,90,40],
                   # 'sex':['boy','girl','girl','boy','girl','boy','girl','girl','boy','girl'],
                   'height':[2340, 2340, 736, 2400, 812, 2340, 2340, 2160, 736, 2160],
                   'width':[1080, 1080, 414, 1080, 375, 1080, 1080, 1080, 414, 1080],
                   'login_cnt_30':[0,10,20,0,11,55,0,6,30,40]
                   })
y=[1,0,1,1,1,0,1,0,0,1]
dump_svmlight_file(df.values,y,'dataset.libsvm',zero_based=False)## 默认为zero_based=True，转换后的字段编号从0开始