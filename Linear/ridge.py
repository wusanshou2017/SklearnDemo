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
    测试 Ridge 的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def ridge(*data):
    X_train,X_test,y_train,y_test=data
    regr=linear_model.Ridge()
    regr.fit(X_train,y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

'''
    测试 Ridge 的预测性能随 alpha 参数的影响
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def ridge_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores=[]
    for i,alpha in enumerate(alphas):
        regr=linear_model.Ridge(alpha=alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    ##绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    plt.show()

if __name__=='__main__':
    X_train, X_test, y_train, y_test = load_data()
    ridge(X_train,X_test,y_train,y_test)
    ridge_alpha(X_train,X_test,y_train,y_test)
