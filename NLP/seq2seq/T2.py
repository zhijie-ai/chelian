#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/23 16:59                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

data  = np.array([[-1,1,0],
                  [-4,3,0],
                  [1,0,2]])

e_vals,e_vecs = np.linalg.eig(data)
print(e_vals)
print(e_vecs)

# 所以λ=2时，特征向量=[0,0,1]
# λ=1时，特征向量=[0.40824829,0.81649658, -0.40824829]
# 这样也就是说，我们可以不用σi=Avi/ui来计算奇异值，也可以通过求出A.TA的特征值取平方根来求奇异值。
print('A'*50)
A=np.array([[0,1],
            [1,1],[1,0]])
A = np.dot(A.T,A)
e_vals,e_vecs = np.linalg.eig(A)
print(e_vals)
print(e_vecs)

import copy
print(copy.deepcopy(A))