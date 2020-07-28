#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/3/27 10:44                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 1.SGAN(最原始的GAN)存在梯度消失的问题，
# 2.WGAN(W距离GAN)，可以解决梯度消失的问题，但引入了一个L约束
# 3.WGAN-gp:将梯度惩罚加到判别器中损失中进行优化
# 4.WGAN的谱归一化:要求网络的每一层都满足L约束，条件太强
# 5.WGAN-div(W散度):https://zhuanlan.zhihu.com/p/49046736,竟然证明了wgan-gp最大化之后不是一个散度，与
#     wgan-gp对着干？
# 6.WGAN-QP(quadratic potential divergence,平方势散度)
#   https://www.jiqizhixin.com/articles/2018-11-27-24
