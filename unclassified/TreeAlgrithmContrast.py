#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/15 16:15
 =================知行合一=============
 树系列算法对比
'''

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


iris = load_iris()
X,y = iris.data,iris.target
X_gen, y_gen = make_classification(50000, n_classes=8, n_clusters_per_class=2, n_informative=4)
# X_train,X_test,y_train,y_test = train_test_split(X_gen,y_gen,test_size=0.2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

def gridSearchCV():
    from sklearn.model_selection import GridSearchCV
    import time

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4)
    param_grid = {'n_estimators': [80, 100, 120],
                  'max_depth': [4, 6, 8],
                  'criterion': ['gini', 'entropy'],
                  }
    start = time.time()
    grid = GridSearchCV(rf_model, param_grid=param_grid, cv=5)
    grid = grid.fit(X_train, y_train)
    end = time.time()
    print('cv_results_:', grid.cv_results_)
    print('best_estimator_', grid.best_estimator_)
    print('best_score_', grid.best_score_)
    print('best_params_', grid.best_params_)
    print('best_index_', grid.best_index_)
    print('scorer_', grid.scorer_)
    print('n_splits_', grid.n_splits_)
    print('总共花费时间：', str(end - start))

def randomForest():
    model = RandomForestClassifier(n_estimators=100,max_depth=6)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('随机森林的acc',accuracy_score(y_test,y_pred))
    print('classification_report', classification_report(y_test, y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))

def adaBoost():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    decisionTree = DecisionTreeClassifier()

    model = AdaBoostClassifier(n_estimators=100,base_estimator=decisionTree)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('adaBoost的acc', accuracy_score(y_test, y_pred))
    print('classification_report', classification_report(y_test, y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))

def gbdt():
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('gbdt的acc', accuracy_score(y_test, y_pred))
    print('classification_report',classification_report(y_test,y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))

#测试标签直接用类别，而不是用索引代替(结果可行,但是不知道那个类别对应哪组参数)
# 不管是ovr还是multinomial，shape都是(3,4)
def LR():
    from sklearn.linear_model import LogisticRegression

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train = X[0:100]
    # X_test = X[100:]
    # y_train = y[0:100]
    # y_test = y[100:]
    model = LogisticRegression(multi_class='multinomial',solver='sag')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('acc:',accuracy_score(y_test,y_pred))
    print('y_test:',y_test)
    print('model.coef_.shape',model.coef_.shape)
    print('set(y_train)',set(y_train))
    print('coef.shape',model.coef_.shape)

def LR2():
    from sklearn.cluster import KMeans

    iris = load_iris()
    X, y = iris.data, iris.target

    model = KMeans(n_clusters=3)
    k = model.fit_transform(X)
    print(model.cluster_centers_)
    print(model.labels_)
    print('y:',y)
    print(k.shape)
    print(model.inertia_)
    from sklearn.model_selection import train_test_split

def xgboost():
    from xgboost.sklearn import XGBClassifier

    model = XGBClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('xgboost的acc', accuracy_score(y_test, y_pred))
    print('classification_report', classification_report(y_test, y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))

def decisionTree():
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('decisionTree的acc', accuracy_score(y_test, y_pred))
    print('classification_report', classification_report(y_test, y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))


#lightgbm
def lightgbm():
    from lightgbm.sklearn import LGBMClassifier

    model = LGBMClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('lightgbm的acc', accuracy_score(y_test, y_pred))
    print('classification_report', classification_report(y_test, y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))

def decorate(func):
    import time

    start = time.time()
    func()
    end = time.time()
    print('{}函数运行的时间为{}'.format(func.__name__,str(end-start)))

if __name__ == '__main__':
    # decorate(decisionTree)
    # decorate(randomForest)
    # decorate(adaBoost)
    # decorate(gbdt)
    # decorate(xgboost)
    # gridSearchCV()
    # LR()
    LR2()


