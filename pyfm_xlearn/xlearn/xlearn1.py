# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/25 21:11                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import xlearn as xl
#
# Training task
ffm_model = xl.create_ffm()
ffm_model.setTrain('../data/small_train.txt')

param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002}
# Train model
ffm_model.setTXTModel('model.txt')
ffm_model.fit(param, './model.out')

# ffm_model.setSigmoid()
ffm_model.setSign()
ffm_model.setTest("../data/small_test.txt")
ffm_model.predict("./model.out", "./output.txt")

# 在线学习
# import xlearn as xl
# ffm_model = xl.create_ffm()
# ffm_model.setTrain("../data/small_train.txt")
# ffm_model.setValidate("../data/small_test.txt")
# ffm_model.setPreModel("./model.out")
# param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002}
#
# ffm_model.fit(param, "./model.out")

# 选择机器学习算法
# 目前，xLearn 可以支持三种不同的机器学习算法，包括了线性模型 (LR)、factorization machine (FM)，以及 field-aware factorization machine (FFM):
# import xlearn as xl
# ffm_model = xl.create_ffm()
# fm_model = xl.create_fm()
# lr_model = xl.create_linear()
# 对于 LR 和 FM 算法而言，我们的输入数据格式必须是 CSV 或者 libsvm.
# 对于 FFM 算法而言，我们的输入数据必须是 libffm 格式: