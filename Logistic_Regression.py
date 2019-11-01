from sklearn.datasets import load_breast_cancer
# import matplotlib.pyplot as plt
import math
import numpy as np
from  sklearn.model_selection import train_test_split
#导入数据
X=np.array(load_breast_cancer().data)
Y=np.array(load_breast_cancer().target)
n=X.shape[0] #特征维数（行数）

#处理数据

X=np.row_stack((np.ones((1,n)),X))#X需要加上一维
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3) #划分训练集和测试集

def sigmoid(Z):
    return 1/(1+math.exp(-Z))

def J(theta): #cost function
    p=sigmoid(np.dot(x_train.T,theta))
    return -np.dot(y_train.T,math.log(p))-np.dot((1-y_train).T,math.log(1-p))

#初始化Θ
theta=np.zeros((n+1,1))
alpha=0.1 #学习率

# 梯度下降
J_last=J(theta)+1 #J_last：上一次J的值    #初始化为J(theta)+1
while math.fabs(J_last-J(theta))>0.001: #迭代停止条件？
    theta=theta - alpha * np.dot(x_train.T,sigmoid(np.dot(x_train.T,theta)))

#预测
def score(X,y,theta):
    pred=np.dot(X.T,theta)
    result=np.array((pred-0.5)*(y-0.5))
    return result[result>0].shape[0]/y.shape[0]

print("train score",score(x_train,y_train,theta))
print("test score",score(x_test,y_test,theta))