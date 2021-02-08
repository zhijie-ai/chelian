#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/4 16:57                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import os, sys
tf_python = 'D:/Software/Python35/python.exe'
exec_file = 'tensorflow_main.py'
if len(sys.argv) > 1:
	exec_file = sys.argv[1]
os.system('%s %s'%(tf_python, exec_file))