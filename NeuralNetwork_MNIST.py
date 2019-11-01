#!/usr/bin/env python
# coding: utf-8

# In[1]:


import struct
import numpy as np
import math
from sklearn.preprocessing import StandardScaler


# In[2]:


# 载入MNIST数据集
def load_MNIST(path,kind):#path和kind是str，kind是train或t10k；函数返回tuple
    with open(path+kind+'-labels.idx1-ubyte','rb') as flabel:#打开label文件
        magic_lb, n_lb = struct.unpack('>II',flabel.read(8)) #read()从文件中读取指定个数字节；unpack()将字节流转化为指定类型（int32),返回tuple
                                                    #文件最开始有两个4字节int分别是magic number（区分文件是图片还是标签）和number of items,
                                                    #'>'大端转小端；'II'指有两个int
        labels = np.fromfile(flabel , dtype=np.uint8) #按照uint8的格式从文件中读取数字，存入labels数组

    #print(magic,n,labels)

    with open(path+kind+'-images.idx3-ubyte','rb') as fimage:#打开image文件
        magic_im, n_im, row, col = struct.unpack('>IIII',fimage.read(16)) #read()从文件中读取指定个数字节；unpack()将字节流转化为指定类型（int32),返回tuple
                                                    #文件最开始有4个4字节int分别是magic number,number of image,number of rows（28）,number of columns（28）
                                                    #'>'大端转小端；'IIII'指有4个int
        images = np.fromfile(fimage , dtype=np.uint8) #按照uint8的格式从文件中读取数字，存入image数组
    return labels , images , n_lb


# In[3]:


y_train, x_train, train_num = load_MNIST('','train')
y_test, x_test, test_num = load_MNIST('','t10k')
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
# print("train_images\n",x_train,"\ntrain_labels\n",y_train,"\ntest_images\n",x_test,"\ntest_labels\n",y_test)
m=train_num# 训练集样本个数
n=784 #特征维数
# x_train.shape,train_num,x_train[0]


# In[4]:


#将label转化为one_hot编码
def one_hot_label(label,m):
    y_temp=np.ndarray([1,1])
    for i in range(m):
        temp=np.zeros([1,10])
        temp[0][label[i]]=1
        if i==0:
            y_temp=temp
        else:
            y_temp=np.vstack([y_temp,temp])
#         print(label[i])
    return y_temp


# In[5]:


#数据处理
#正规化 将像素取值压缩到[0-1] #???可能出错
x_train=x_train/256
x_test=x_test/256
# x_train[0].shape

#将label转化为one_hot编码
# y_train=one_hot_label(y_train,train_num)
y_test=one_hot_label(y_test,test_num)
y_train,y_test


# In[6]:


#激活函数
def sigmoid(Z):
    return 1/(1+np.exp(-1*Z)+ 1e-5)


# In[7]:


#初始化一个a*b的矩阵
def init_matrix(a,b): 
    v=np.zeros([a,b])
    for i in range (a):
        for j in range (b):
            v[i][j]=np.random.uniform(0,1)
    return v


# In[8]:


def forward_propagation(x_test,v,w,gama,theta):
    x=x_test.reshape(-1,1) #n*1
    #前馈计算，输出为a
    alpha=(np.dot(x.T,v)).T #隐层输入，h_size*1
    b=sigmoid(alpha-gama) #隐层输出，h_size*1
    beta=(np.dot(b.T,w)).T #输出层输入，o_size*1
    a=sigmoid(beta-theta) #输出层输出，o_size*1
    
    maxP=0 #最大概率
    maxI=-1 #最大下标
    for i in range(10):# 找到最大概率，作为预测结果
        if a[i][0]>maxP:
            maxP=a[i][0]
            maxI=i
    ret=np.zeros([10,1])
    ret[i][0]=1
    return ret.T


# In[9]:


def score(x_test,y_test,m,v,w,gama,theta):
    cnt=0
    for i in range(m):
        result=forward_propagation(x_test[i],v,w,gama,theta)
        if (result==y_test[i]).all():
            cnt=cnt+1
    print(cnt,m)
    return cnt/m


# In[10]:


eta=3 #学习率
h_size= 300 #hidden layer神经元个数
o_size=10 #输出单元个数

v=init_matrix(n,h_size)#input layer与hiddenlayer连接权
w=init_matrix(h_size,o_size)
gama=init_matrix(h_size,1)
theta=init_matrix(o_size,1)
v,w,gama,theta


# In[11]:


def BP(x_train,y_train,m,v,w,gama,theta):
#     preE=22
#     E=20
    iterCnt=0
    maxIter=200
#     while(math.fabs(preE-E)>0.0001):
    while(maxIter>0):
        maxIter=maxIter-1
        iterCnt=iterCnt+1
        print(iterCnt)
#         preE=E
        for i in range(m):
            x=(x_train[i]).reshape(-1,1) #n*1
            y=(y_train[i]).reshape(-1,1) #o_size*1 ?????
            #前馈计算，输出为a
            alpha=(np.dot(x.T,v)).T #隐层输入，h_size*1
            b=sigmoid(alpha-gama) #隐层输出，h_size*1
            beta=(np.dot(b.T,w)).T #输出层输入，o_size*1
            a=sigmoid(beta-theta)
            
#             print(x.shape,y.shape,alpha.shape,b.shape,beta.shape,a.shape)

            #BP
            g=a*(1-a)*(y-a) #对应元素相乘，得到o_size * 1的矩阵
            e=b*(1-b)*np.dot(w,g)

            w=w+eta*np.dot(b,g.T) #h_size*o_size
            theta= theta-eta*g
            v=v+eta*np.dot(x,e.T)
            gama=gama-eta*e
#         E=(1/m)*np.dot(a-y,a-y)
        print(score(x_test,y_test,test_num,v,w,gama,theta),theta)
    print(iterCnt)
    
    return v,w,gama,theta


# In[ ]:


# v,w,gama,theta=BP(x_train,y_train,m,v,w,gama,theta)
v,w,gama,theta=BP(x_test,y_test,test_num,v,w,gama,theta)
v,w,gama,theta


# In[ ]:


# score(x_train,y_train,train_num,v,w,gama,theta),score(x_test,y_test,test_num,v,w,gama,theta)
print("train set score：",score(x_train,y_train,train_num,v,w,gama,theta))
print("test set score：",score(x_test,y_test,test_num,v,w,gama,theta))

# In[ ]:




