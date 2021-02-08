#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/9 9:54
 =================知行合一=============
'''



#移除方差较低的特征
from sklearn.ensemble.iforest import IsolationForest


def varianceThreshold():
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import VarianceThreshold
    iris = load_iris()
    X,y = iris.data,iris.target
    ##设置方差的阈值为0.8
    model = VarianceThreshold(threshold=0.08)
    # 选择方差大于0.8的特征
    X_new = model.fit_transform(X,y)
    print(X_new.shape)


#单变量特征选择(Univariate feature selection),
# 除了使用SelectKBest之外，还可以使用SelectPercentile，她是按百分比进行选择的。
# 回归问题:f_regression, mutual_info_regression
# 分类问题:chi2, f_classif, mutual_info_classif
def univariateFeatureSelection():
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_regression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import  ExtraTreeClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.utils import shuffle

    iris = load_iris()
    X,y = iris.data,iris.target
    print(X.shape)
    X_new = SelectKBest(chi2,k=2).fit_transform(X,y)
    print(X_new.shape)

def univariateFeatureSelection2():
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_regression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import  ExtraTreeClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.utils import shuffle
    from sklearn.preprocessing import  StandardScaler
    from sklearn.model_selection import learning_curve

    import numpy as np
    import matplotlib.pyplot as plt

    iris = load_iris()
    # 添加（150，20）的随机噪声
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    X = np.hstack((iris.data,E))
    y = iris.target
    print(X.shape)
    X_indices = np.arange(X.shape[-1])
    selector = SelectPercentile(f_classif,percentile=10)
    selector.fit(X,y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)', color='g')
    plt.show()
    X_new = selector.transform(X)
    print(X_new.shape)


    def testNormalDistribution():
        import matplotlib.mlab as mlab
        mu, sigma, num_bins = 0, 1, 50
        x = mu + sigma * np.random.randn(1000000)
        # 正态分布的数据
        n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='blue', alpha=0.5)
        # 拟合曲线
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        plt.xlabel('Expectation')
        plt.ylabel('Probability')
        plt.title('histogram of normal distribution: $\mu = 0$, $\sigma=1$')

        plt.subplots_adjust(left=0.15)
        plt.show()

    def testNormalDistribution2():
        data = np.random.randn(100000)
        plt.hist(data,bins=60)
        plt.show()

    testNormalDistribution2()
# 基于学习模型的特征排序
def modelBasedRanking():
    from sklearn.cross_validation import cross_val_score, ShuffleSplit
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    from sklearn.linear_model import LinearRegression
    boston = load_boston()
    X = boston['data']
    y = boston['target']
    names = boston['feature_names']

    rf = LinearRegression()
    scores = []

    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i + 1], y, scoring='r2',
                                cv=ShuffleSplit(len(X), 3, 3))
        scores.append((round(np.mean(score), 3), names[i]))

    print(sorted(scores, reverse=True))
    # print(rf.feature_importances_)
    # print(rf.n_features_)

# 递归特征消除Recursive feature elimination （RFE）
def rfe():
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt

    digits = load_digits()
    X = digits.images.reshape(len(digits.images),-1)
    y = digits.target

    #Create the RFE object and rank each pixel
    svc = SVC(kernel='linear',C=1)
    rfe = RFE(estimator=svc,n_features_to_select=1,step=1)
    rfe.fit(X,y)
    ranking = rfe.ranking_.reshape(digits.images[0].shape)
    print(ranking.shape)

    plt.matshow(ranking)
    plt.colorbar()
    plt.title('Ranking of pixels with RFE')
    plt.show()



#基于L1的特征选择(L1-based feature selection)
def selectFromModelL1Based():
    from sklearn.svm import LinearSVC
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    iris = load_iris()
    X,y = iris.data,iris.target
    print(X.shape)
    # lsvc = LinearSVC(C=0.01,penalty='l1',dual=False).fit(X,y)
    lr = LogisticRegression(penalty='l1',C=0.1).fit(X,y)
    model = SelectFromModel(lr,prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)

#基于树的特征选择(Tree-based feature selection)
def selectFromModelTreeBased():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    X,y = iris.data,iris.target
    print(X.shape)
    clf = ExtraTreesClassifier()
    clf = DecisionTreeClassifier(max_depth=4)
    # clf = clf.fit(X,y)
    # print(clf.feature_importances_)
    model = SelectFromModel(clf)
    model.fit(X,y)
    X_new = model.transform(X)
    print(X_new.shape)

#基于树的特征选择(Tree-based feature selection)2,应该和selectFromModelTreeBased一样。
def selectFromModelTreeBased2():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import  SelectFromModel

    iris = load_iris()
    X,y = iris.data,iris.target
    print(X.shape)
    clf = DecisionTreeClassifier(max_depth=4)# 不需要先fit，可以直接调用sfm.fit(X,y)，能成功
    # clf.fit(X,y)
    # clf = ExtraTreesClassifier(max_depth=4)# 不需要先fit，可以直接调用sfm.fit(X,y)，能成功
    sfm = SelectFromModel(clf)
    sfm.fit(X,y)
    X_new = sfm.transform(X)
    print(X_new.shape)

#基于树的特征选择(Tree-based feature selection)3,应该和selectFromModelTreeBased一样。
def selectFromModelTreeBased3():
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_boston
    from sklearn.feature_selection import  SelectFromModel
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    iris = load_boston()
    X,y = iris.data,iris.target
    # clf = LinearRegression()#不先训练，直接调用sfm.fit是可以成功的
    # clf = RandomForestRegressor(max_depth=4) #不先训练，直接调用sfm.fit是可以成功的
    clf = RandomForestRegressor(max_depth=4) #不先训练，直接调用sfm.fit是可以成功的
    sfm = SelectFromModel(clf)
    sfm.fit(X,y)
    X_new = sfm.transform(X)
    print(X_new.shape)

def selectFromModelLasso():
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.datasets import load_boston
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LarsCV

    boston = load_boston()
    X,y = boston.data,boston.target
    clf = LarsCV()

    sfm = SelectFromModel(clf,threshold=0.25)
    sfm.fit(X,y)
    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features>2:
        sfm.threshold +=0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    # Plot the selected two features from X.
    plt.title("Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
    feature1 = X_transform[:,0]
    feature2 = X_transform[:,1]
    plt.plot(feature1,feature2,'r.')
    plt.xlabel('Feature number 1')
    plt.xlabel('Feature number 2')
    plt.ylim([np.min(feature2),np.max(feature2)])
    plt.show()

# 用svm做异常值的检测
def noise_detection():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    from sklearn import svm

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()

if __name__ == '__main__':
    # varianceThreshold()
    # univariateFeatureSelection()
    # univariateFeatureSelection2()
    # selectFromModelL1Based()
    # selectFromModelTreeBased()
    # selectFromModelTreeBased2()
    # rfe()
    # selectFromModelLasso()
    # selectFromModelTreeBased3()
    # modelBasedRanking()
    # noise_detection()
    from sklearn.linear_model import RandomizedLogisticRegression
    from sklearn.linear_model import LogisticRegression
    LR_moel = LogisticRegression
    LR_moel.score()
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
