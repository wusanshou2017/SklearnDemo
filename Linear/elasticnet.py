#!/usr/bin/python
#-*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

'''
加载用于回归问题的数据集
元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
'''
def load_data():
    diabetes=datasets.load_diabetes()
    return train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)
'''
    测试 ElasticNet 的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def elasticnet(*data):
    X_train,X_test,y_train,y_test=data
    regr=linear_model.ElasticNet()
    regr.fit(X_train,y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

'''
    测试 ElasticNet 的预测性能随 alpha ,rhos参数的影响
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def elasticnet_alpha_rho(*data):
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,2)
    rhos=np.linspace(0.01,1)
    scores=[]
    for alpha in alphas:
            for rho in rhos:
                regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)
                regr.fit(X_train, y_train)
                scores.append(regr.score(X_test, y_test))
    ## 绘图
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig=plt.figure()
    ax=Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()

if __name__=='__main__':
    X_train, X_test, y_train, y_test = load_data()
    elasticnet(X_train,X_test,y_train,y_test)
    elasticnet_alpha_rho(X_train,X_test,y_train,y_test)