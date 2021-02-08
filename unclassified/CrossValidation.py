#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/9 17:42
 =================知行合一=============
'''

# cross_val_score
# 对数据集进行指定次数的交叉验证并为每次验证效果评测
# score 默认是以 scoring=’f1_macro’进行评测的，余外针对分类或回归还有
# 这需要from　sklearn import metrics ,通过在cross_val_score 指定参数来设定评测标准；
# 当cv 指定为int 类型时，默认使用KFold 或StratifiedKFold 进行数据集打乱，
# 下面会对KFold 和StratifiedKFold 进行介绍
from sklearn.externals.joblib.parallel import Parallel


def cross_val_score():
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import  load_iris
    from sklearn.svm import SVC
    import sklearn

    iris = load_iris()

    clf = SVC(kernel='linear',C=1)
    scores = cross_val_score(clf,iris.data,iris.target,cv=5)#多分类不支持,scoring='roc_auc'
    print(sklearn.metrics.SCORERS.keys())
    print(scores)
    print('means',scores.mean())

# 除使用默认交叉验证方式外，可以对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
def cross_val_score2():
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import  load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import ShuffleSplit

    iris = load_iris()
    n_samples = iris.data.shape[0]

    cv = ShuffleSplit(n_splits=3,test_size=.3,random_state=0)
    clf = SVC(kernel='linear',C=1)
    scores = cross_val_score(clf,iris.data,iris.target,cv=cv)
    print(scores)
    print('means',scores.mean())

# 在cross_val_score 中同样可使用pipeline 进行流水线操作
def cross_val_score3():
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import  load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import ShuffleSplit
    from sklearn import preprocessing
    from sklearn.pipeline import  Pipeline
    from sklearn.pipeline import Parallel
    from sklearn.pipeline import make_pipeline

    iris = load_iris()
    n_samples = iris.data.shape[0]

    cv = ShuffleSplit(n_splits=3,test_size=.3,random_state=0)
    clf = make_pipeline(preprocessing.StandardScaler(),SVC(C=1))
    scores = cross_val_score(clf,iris.data,iris.target,cv=cv)
    print(scores)
    print('means',scores.mean())

'''
cross_val_predict 与cross_val_score 很相像，不过不同于返回的是评测效果，
cross_val_predict 返回的是estimator 的分类结果（或回归值），
这个对于后期模型的改善很重要，可以通过该预测输出对比实际目标值，
准确定位到预测出错的地方，为我们参数优化及问题排查十分的重要。
'''
def cross_val_predict():
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC

    iris = load_iris()
    clf = SVC(kernel='linear',C=1)
    predicted = cross_val_predict(clf,iris.data,iris.target,cv=10)
    print('accuracy:',metrics.accuracy_score(iris.target,predicted))
    print(predicted)


# cross_validate
# cross_validate方法和cross_validate_score有个两个不同点：
# 它允许传入多个评估方法，可以使用两种方法来传入，一种是列表的方法，
# 另外一种是字典的方法。最后返回的scores为一个字典，字典的key为：
# dict_keys(['fit_time', 'score_time', 'test_score', 'train_score'])
# 下面是它的演示代码，当scoring传入列表的时候如下
def cross_validate():
    from sklearn.model_selection import cross_validate
    from sklearn.svm import  SVC
    from sklearn.datasets import load_iris

    iris = load_iris()
    # 如果只有一个score的话，返回的key为dict_keys(['test_precision_macro', 'fit_time', 'score_time'])
    scoring = ['precision_macro','recall_macro']
    clf = SVC(kernel='linear',C=1,random_state=0)
    scores = cross_validate(clf,iris.data,iris.target,scoring=scoring,cv=5,return_train_score=False)
    print(scores.keys())
    print(scores['test_recall_macro'])

# 当scoring传入字典的时候如下：
def cross_validate2():
    from sklearn.model_selection import cross_validate
    from sklearn.svm import  SVC
    from sklearn.metrics import make_scorer,recall_score
    from sklearn.datasets import load_iris

    iris = load_iris()
    scoring = {'prec_macro':'precision_macro','rec_micro':make_scorer(recall_score,average='macro')}
    clf = SVC(kernel='linear',C=1,random_state=0)
    scores = cross_validate(clf,iris.data,iris.target,scoring=scoring,cv=5,return_train_score=False)
    print(scores.keys())
    print(scores['test_rec_micro'])



# K折交叉验证，这是将数据集分成K份的官方给定方案，所谓K折就是将数据集通过K次分割，
# 使得所有数据既在训练集出现过，又在测试集出现过，当然，每次分割中不会有重叠。相当于无放回抽样。
def kfold():
    from sklearn.model_selection import KFold
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X= ['a','b','c','d']
    kf = KFold(n_splits=2)
    for train,test in kf.split(X):
        print(train,test)
        print(np.array(X)[train], np.array(X)[test])
        print()

# RepeatedKFold  p次k折交叉验证
def repeatedKFold():
    from sklearn.model_selection import RepeatedKFold
    import numpy as np

    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])
    kf = RepeatedKFold(n_splits=2,n_repeats=2,random_state=0)
    for train_index,test_index in kf.split(X):
        print('train_index',train_index,'test_index',test_index)


# LeaveOneOut 留一法
# 留一法的缺点是：当n很大的时候，计算量会很大，因为需要进行n次模型的训练，
# 而且训练集的大小为n-1。建议k折交叉验证的时候k的值为5或者10。
def leaveOneOut():
    from sklearn.model_selection import LeaveOneOut
    X = [1,2,3,4]
    loo = LeaveOneOut()
    for train_index,test_index in loo.split(X):
        print('train_index', train_index, 'test_index', test_index)

# LeavePOut  留P法
def leavePOut():
    from sklearn.model_selection import LeavePOut
    X = [1, 2, 3, 4]
    loo = LeavePOut(p=2)
    for train_index, test_index in loo.split(X):
        print('train_index', train_index, 'test_index', test_index)

# ShuffleSplit  随机分配
# 使用ShuffleSplit方法，可以随机的把数据打乱，然后分为训练集和测试集。
# 它还有一个好处是可以通过random_state这个种子来重现我们的分配方式，如果没有指定，那么每次都是随机的。
def shuffleSplit():
    from sklearn.model_selection import ShuffleSplit
    import numpy as np

    X = np.arange(5)
    ss = ShuffleSplit(n_splits=4,random_state=0,test_size=0.25)
    for train_index, test_index in ss.split(X):
        print('train_index', train_index, 'test_index', test_index)

# 其它特殊情况的数据划分方法
# 1：对于分类数据来说，它们的target可能分配是不均匀的，比如在医疗数据当中得癌症的人比不得癌症的人少很多，
# 这个时候，使用的数据划分方法有  StratifiedKFold  ，StratifiedShuffleSplit
# 2：对于分组数据来说，它的划分方法是不一样的，主要的方法有 GroupKFold，LeaveOneGroupOut
# ，LeavePGroupOut，GroupShuffleSplit
# 3：对于时间关联的数据，方法有TimeSeriesSplit

if __name__ == '__main__':
    # cross_val_score()
    # cross_val_score2()
    # cross_val_predict()
    # kfold()
    # repeatedKFold()
    # leaveOneOut()
    # leavePOut()
    # shuffleSplit()
    # cross_validate()
    cross_validate2()