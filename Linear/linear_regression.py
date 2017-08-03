#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
'''
加载用于回归问题的数据集
元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
'''
diabetes = datasets.load_diabetes()#使用 scikit-learn 自带的一个糖尿病病人的数据集
X=diabetes.data
y=diabetes.target
X_train, X_test, y_train, y_test=train_test_split(X,y,
		test_size=0.25,random_state=0) # 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
def LinearRegression(*data):
    '''
    测试 LinearRegression 的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))
if __name__=='__main__':
    LinearRegression(X_train,X_test,y_train,y_test) # 调用 test_LinearRegression
