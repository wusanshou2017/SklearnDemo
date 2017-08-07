#!/usr/bin/python
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split

'''
加载用于回归问题的数据集
元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
'''
def load_data():
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)
'''
    测试LogisticRegression的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def LogisticRegression(*data):
    X_train,X_test,y_train,y_test=data
    regr=linear_model.LogisticRegression()
    regr.fit(X_train,y_train)
    print('Coefficients:%s,intercept %s'%(regr.coef_,regr.intercept_))
    print('Score: %.2f'%regr.score(X_test,y_test))
'''
    测试LogisticRegression的预测性能随 multi_class 参数的影响
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :multi_class:指定对于多分类问题的策略。'ovr':one-vs-rest,'multinomial':直接采用多分类逻辑回归策略
    :solver:指定求解最优化问题的算法。'newton-cg':牛顿法，'lbfgs':,拟牛顿法，'liblinear':liblinear.
    :solver为牛顿法和拟牛顿法才能配合multi_class='multinomial'使用，否则会报错
    :return: None
'''
def LogisticRegression_multinomial(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients:%s,intercept %s' % (regr.coef_, regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))
'''
    测试 LogisticRegression的预测性能随 C 参数的影响
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :C:指定了惩罚系数的倒数，如果它的值越小，则正则化项越大。
    :return: None
'''
def LogisticRegression_C(*data):
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-2,4,num=100)
    scores=[]
    for C in Cs:
        regr=linear_model.LogisticRegression(C=C)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    ##绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LogisticRegression")
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    #LogisticRegression(X_train,X_test,y_train,y_test)
    #LogisticRegression_multinomial(X_train, X_test, y_train, y_test)
    LogisticRegression_C(X_train,X_test,y_train,y_test)







































