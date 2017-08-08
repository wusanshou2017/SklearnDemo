#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
'''
    产生用于回归问题的数据集
    :param n: 数据集容量
    :return: 返回一个元组，元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
'''
def create_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3 * (0.5-np.random.rand(noise_num)) #以5为间隔添加噪音
    return train_test_split(X,y,test_size=0.25,random_state=1)
'''
    测试DecisionTreeRegressor的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def tDecisionTreeRegressor(*data):
     X_train,X_test,y_train,y_test = data
     regr = DecisionTreeRegressor()
     regr.fit(X_train,y_train)
     print("Training score:%f"%(regr.score(X_train,y_train)))
     print("Testing score:%f"%(regr.score(X_test,y_test)))
     ##绘图
     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)
     X = np.arange(0.0,5.0,0.01)[:, np.newaxis]
     Y = regr.predict(X)
     ax.scatter(X_train,y_train,label="train sample",c='g')
     ax.scatter(X_test, y_test, label="test sample", c='r')
     ax.plot(X,Y,label="predict_value",linewidth=2,alpha=0.5)
     ax.set_xlabel("data")
     ax.set_xlabel("target")
     ax.set_title(" Decision Tree Regressor")
     ax.legend(framealpha=0.5)
     plt.show()
'''
    测试DecisionTreeRegressor的预测性能随splitter参数的影响
    :splitter:指定切分原则。'best':选择最优的切分，'random':随机切分
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def DecisionTreeRegressor_splitter(*data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best','random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train,y_train)
        print("Splitter %s"%splitter)
        print("Training score:%f" % (regr.score(X_train, y_train)))
        print("Testing score:%f" % (regr.score(X_test, y_test)))
'''
    测试DecisionTreeRegressor的预测性能随max_depth参数的影响
    :max_depth:可以为整数或None，指定树的最大深度。如果为None则表示深度不限（直到每个叶子都是纯的）。
    :max_leaf_nodes:可以为整数或None，指定叶节点的最大数量。如果为None则表示数量不限，若为非None，则max_depth被忽略。
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test,maxdepth):
    depths = np.arange(1,maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train,y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
    ##绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="training_scores")
    ax.plot(depths, testing_scores, label="testing_scores")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()
'''
    测试DecisionTreeRegressor的预测性能随max_depth参数的影响
    :max_depth:可以为整数或None，指定树的最大深度。如果为None则表示深度不限（直到每个叶子都是纯的）。
    :max_leaf_nodes:可以为整数或None，指定叶节点的最大数量。如果为None则表示数量不限，若为非None，则max_depth被忽略。
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def tDecisionTreeRegressor_depth(*data):
    X_train, X_test, y_train, y_test = data
    regr = DecisionTreeRegressor(max_depth=7)
    regr.fit(X_train,y_train)
    print("Depth %s" % 7)
    print("Training score:%f" % (regr.score(X_train, y_train)))
    print("Testing score:%f" % (regr.score(X_test, y_test)))
    ##绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = regr.predict(X)
    ax.scatter(X_train, y_train, label="train sample", c='g')
    ax.scatter(X_test, y_test, label="test sample", c='r')
    ax.plot(X, Y, label="predict_value:max_depth=7", linewidth=2, alpha=0.5)
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()


if __name__ == '__main__':
    X_train,X_test,y_train,y_test = create_data(100)
    #tDecisionTreeRegressor(X_train,X_test,y_train,y_test)
    #DecisionTreeRegressor_splitter(X_train,X_test,y_train,y_test)
    #DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test,maxdepth=20)
    tDecisionTreeRegressor_depth(X_train,X_test,y_train,y_test)
