#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,tree
from sklearn.model_selection import train_test_split
'''
加载用于决策树分类的数据集
元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
'''
def load_data():
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)
'''
    测试DecisionTreeClassifier的用法
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def DecisionTreeClassifier(*data):
    X_train,X_test,y_train,y_test=data
    clf=tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print("Training score:%f"%(clf.score(X_train,y_train)))
    print("Testing score:%f"%(clf.score(X_test,y_test)))
'''
    测试DecisionTreeClassifier的分类性能随criterion参数的影响
    :criterion:指定切分质量的评价准则。'gini':gini系数，'entropy':熵
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def DecisionTreeClassifier_criterion(*data):
    X_train,X_test,y_train,y_test=data
    criterions=['gini','entropy']
    for criterion in criterions:
        clf=tree.DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train,y_train)
        print("Criterion:%s"%criterion)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
'''
    测试DecisionTreeClassifier的分类性能随splitter参数的影响
    :splitter:指定切分原则。'best':选择最优的切分，'random':随机切分
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def DecisionTreeClassifier_splitter(*data):
    X_train,X_test,y_train,y_test=data
    splitters=['best','random']
    for splitter in splitters:
        clf=tree.DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train,y_train)
        print("Splitter:%s"%splitter)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
'''
    测试DecisionTreeClassifier的预测性能随max_depth参数的影响
    :max_depth:可以为整数或None，指定树的最大深度。如果为None则表示深度不限（直到每个叶子都是纯的）。
    :max_leaf_nodes:可以为整数或None，指定叶节点的最大数量。如果为None则表示数量不限，若为非None，则max_depth被忽略。
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
'''
def DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth):
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_score=[]
    for depth in depths:
        clf=tree.DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_score.append(clf.score(X_test,y_test))
    ##绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="training score",marker='o')
    ax.plot(depths,testing_score,label="testing score",marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()
if __name__ =='__main__':
    X_train,X_test,y_train,y_test=load_data()
    #DecisionTreeClassifier(X_train,X_test,y_train,y_test)
    #DecisionTreeClassifier_criterion(X_train,X_test,y_train,y_test)
    #DecisionTreeClassifier_splitter(X_train,X_test,y_train,y_test)
    DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,20)
