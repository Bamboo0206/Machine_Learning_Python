

# In[1]:


import struct
import numpy as np
from sklearn.preprocessing import StandardScaler

# In[4]:


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


# In[13]:


train_y, train_x, train_num = load_MNIST('','train')
test_y, test_x, test_num = load_MNIST('','t10k')
train_x=train_x.reshape(60000,784)
test_x=test_x.reshape(10000,784)
print("train_images\n",train_x,"\ntrain_labels\n",train_y,"\ntest_images\n",test_x,"\ntest_labels\n",test_y)
m=train_num# 训练集样本个数
n=784 #特征维数
train_x.shape,train_num,train_x[0]


# In[14]:


#数据处理
#正规化 将像素取值压缩到[0-1] #???可能出错
# train_x=train_x/256
# test_x=test_x/256
# # train_x[0].shape
#feature scaling  （特征缩放能使梯度下降更快收敛）
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)

# In[16]:


from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(activation='logistic',hidden_layer_sizes=(300,),verbose=True)
clf.fit(train_x,train_y)#跑了十多分钟。。


# In[ ]:


print("train set score",clf.score(train_x,train_y))
print("test set score",clf.score(test_x,test_y))
