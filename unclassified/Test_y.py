#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/9/20 20:02
 =================知行合一=============
'''

#把y当成一个X的特征，模型会学到以y这个特征直接做决定
def classification():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    iris = load_iris()
    X=iris['data']
    y = iris['target']
    y_ = y.reshape(-1, 1)
    print('X.shape',X.shape)
    print(y_.shape)
    # X = np.concatenate((X,y_),axis=1)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    LR_model = LogisticRegression(penalty='l2',C=0.8)
    LR_model.fit(X_train,y_train)
    score = LR_model.score(X_test,y_test)
    LR_model.decision_function()
    print(score)
    print(LR_model.coef_)

def regression():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import numpy as np

    iris = load_boston()
    X = iris['data']
    y = iris['target']
    y_ = y.reshape(-1, 1)
    print('X.shape', X.shape)
    print(y_.shape)
    X = np.concatenate((X,y_),axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    LR_model = LinearRegression()#penalty='l2', C=0.8
    LR_model.fit(X_train, y_train)
    score = LR_model.score(X_test, y_test)
    print(score)
    print(LR_model.coef_)

if __name__ == '__main__':
    classification()
    # regression()

    from sklearn.svm import NuSVC
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cluster import KMeans

    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingClassifier