import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#生成正态分布数据
def generate_sample(x1,x2): #正样本或负样本，均值坐标，
    N = 100
    x1 = np.random.randn(N) + x1
    x2 = np.random.randn(N) + x2
    x = np.array([x1, x2])
    x = x.T

    return x #100*2向量

#生成数据
data=np.vstack((generate_sample(0,0),generate_sample(3,3),generate_sample(3,-1)))

#画散点图
plt.scatter(data[:,0],data[:,1])
plt.show()

#K-Means
clf = KMeans(n_clusters = 3)#初始化
clf.fit(data)#训练模型
y_pred=clf.predict(data)#预测
# print(y_pred)

#画图，点的颜色使用y_pred
plt.scatter(data[:,0] , data[:,1] , c=y_pred)
plt.show()