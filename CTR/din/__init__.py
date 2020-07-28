#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/12 12:33                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import sys
import os
import json
print(os.getcwd())
# sys.path.insert(0,os.path.dirname(os.getcwd()))
print(json.dumps(sys.path,indent=4))

# utils下面的1_convert_pd.py和2_remap_id.py和convert_pd.py，remap.py文件代码一致

# 1.执行1_convert_pd.py文件
# 2.执行2_remap_id.py文件
# 3.执行build_dataset.py文件，生成训练集及其他数据
# 4.
