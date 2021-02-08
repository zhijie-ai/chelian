#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/30 13:53                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://github.com/guoday/Tencent2019_Preliminary_Rank1st
# 数据预处理 src/preprocess.py
# 提取特征 src/extraction_feature.py
# 转换数据格式 src/convert_format.py
# 训练模型 train.py

# 这种方式似乎没有时间窗口的特征在里面，因为在计算y值(即imp的时候，
#   根据'aid', 'request_day'分组得到)，参考preprocess.py 100行
# 数据源主要有3个文件，用户信息，广告信息(静态和动态，所谓动态，就是修改了某些特征，比如定向特征，cpc特征)，曝光日志。