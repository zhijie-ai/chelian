#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/10/19 10:36                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import sys
import json

print(json.dumps(sys.path,indent=4))

da = {'name':'xiaojie','age':20}
print(type(da))
print(type(json.dumps(da)))