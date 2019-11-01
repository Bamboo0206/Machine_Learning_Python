import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#生成正态分布数据
def generate_sample(x1,x2): #正样本或负样本，均值坐标，
    N = 100
    x1 = np.random.randn(N) + x1
    x2 = np.random.randn(N) + x2
    x = np.array([x1, x2])
    x = x.T
    #print(x.shape)

    return x



#生成数据
positive=generate_sample(0,0) #100*2向量
negative=generate_sample(4,4)
# tempx=positive
# tempy=positive
# print(tempx[:,0].shape,tempy[:,1].shape)
# plt.scatter(tempx[:,0],tempy[:,1])
# tempx=negative
# tempy=negative
# plt.scatter(tempx[:,:1],tempy[:,1:],c="red")#画图
plt.scatter(positive[:,0],positive[:,1])
plt.scatter(negative[:,:1],negative[:,1:],c="red")
plt.show()
#将数据分为训练集和测试集
# tempp=positive
# tempn=negative
# X_train=np.vstack([tempp[:70,:],tempn[:70,:]])
# y_train=np.array([1]*70+[0]*70)
X_train=np.vstack([positive[:70,:],negative[:70,:]])
y_train=np.array([1]*70+[0]*70)
# print(X_train,y_train)

# tempp=positive
# tempn=negative
# X_test=np.vstack([tempp[70:,:],tempn[70:,:]])
# y_test=np.array([1]*30+[0]*30)
X_test=np.vstack([positive[70:,:],negative[70:,:]])
y_test=np.array([1]*30+[0]*30)



#将数据分为训练集和测试集

#训练
svm = SVC(kernel='linear', C=1.0)#使用线性核函数，C=1
svm.fit(X_train,y_train)#拟合模型

print("suport_vectors_:",svm.support_vectors_)#打印出支持向量
print("coef_:",svm.coef_)                  #打印出权重系数，还是w

print("预测测试集准确率：",svm.score(X_test,y_test))

#根据模型拟合出的直线取两个点画决策边界
k = - svm.coef_[0][0] / svm.coef_[0][1] #计算斜率
b = - svm.intercept_[0] / svm.coef_[0][1] #计算截距
x_list=[-3,8] #计算两个直线上的点：取两个x（这两个点的取值需要根据正负样本的质心相应改动）
y_list=[k * x_list[0] + b , k * x_list[1] + b] #计算x对应坐标y
plt.plot(x_list,y_list,'k-')#画决策边界，‘k-’是实线

#画散点图
plt.scatter(positive[:,0],positive[:,1])
plt.scatter(negative[:,:1],negative[:,1:],c="red")

plt.axis('tight')
plt.show()
