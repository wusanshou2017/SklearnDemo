#!/usr/bin/python
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,discriminant_analysis
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
    测试LinearDiscriminationAnalysis的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def LinearDiscriminationAnalysis(*data):
    X_train,X_test,y_train,y_test=load_data()
    lda=discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train,y_train)
    print('Coefficients:%s,intercept %s'%(lda.coef_,lda.intercept_))
    print('Score:%.2f' % lda.score(X_test,y_test))
'''
绘制LDA降维之后的数据集
'''
def plot_LDA(converted_X,y):
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure()
    ax=Axes3D(fig)
    colors='rgb'
    markers='o*s'
    for target,color,marker in zip([0,1,2],colors,markers):
        pos=(y==target).ravel()
        X=converted_X[pos,:]
        ax.scatter(X[:,0],X[:,1],X[:,2],color=color,marker=marker,label="Label %d"%target)
    ax.legend(loc="best")
    fig.suptitle("Iris After LDA")
    plt.show()
'''
执行 plot_LDA 。其中数据集来自于 load_data() 函数
:return: None
'''
def run_plot_LDA(*data):
    X_train,X_test,y_train,y_test=data
    X=np.vstack((X_train,X_test))
    Y=np.vstack((y_train.reshape(y_train.size,1),y_test.reshape(y_test.size,1)))
    lda=discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X,Y)
    converted_X=np.dot(X,np.transpose(lda.coef_))+lda.intercept_
    plot_LDA(converted_X,Y)
'''
    测试LinearDiscriminationAnalysis的预测性能随 solver 参数的影响
    :solver:指定求解最优化问题的算法。
    :'svd':奇异值分解，适合于有大规模特征的数据
    :'lsqr':最小平方差算法，'eigen':特征值分解算法，可以结合shrinkage参数
    :return: None
'''
def LinearDiscriminationAnalysis_solver(*data):
    X_train, X_test, y_train, y_test = data
    solvers=['svd','lsqr','eigen']
    for solver in solvers:
        if(solver=='svd'):
            lda=discriminant_analysis.LinearDiscriminantAnalysis(solver=solver)
        else:
            lda=discriminant_analysis.LinearDiscriminantAnalysis(solver=solver,shrinkage=None)
        lda.fit(X_train,y_train)
        print('Score at solver=%s: %.2f' %(solver,lda.score(X_test,y_test)))
'''
    测试LinearDiscriminationAnalysis的预测性能随 shrinkage 参数的影响
    :shrinkage:字符串'auto'或者浮点数或者None。
    :通常在训练样本数量小于特征数量的场合下使用，引入抖动相当于引入正则化项
    :solver='lsqr'，'eigen'时shrinkage参数才有意义 
    :return: None
'''
def LinearDiscriminationAnalysis_shrinkage(*data):
    X_train, X_test, y_train, y_test = data
    shrinkages=np.linspace(0.0,1.0,num=20)
    scores=[]
    for shrinkage in shrinkages:
        lda=discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',shrinkage=shrinkage)
        lda.fit(X_train,y_train)
        scores.append(lda.score(X_test,y_test))
    ##绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(shrinkages,scores)
    ax.set_xlabel(r"shrinkage")
    ax.set_ylabel(r"score")
    ax.set_ylim(0,1.05)
    ax.set_title("LinearDiscriminationAnalysis")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    #LinearDiscriminationAnalysis(X_train,X_test,y_train,y_test)
    #run_plot_LDA(X_train,X_test,y_train,y_test)
    #LinearDiscriminationAnalysis_solver(X_train,X_test,y_train,y_test)
    LinearDiscriminationAnalysis_shrinkage(X_train,X_test,y_train,y_test)
