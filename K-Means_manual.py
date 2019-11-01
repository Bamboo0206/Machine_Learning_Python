import numpy as np
import matplotlib.pyplot as plt
from tkinter import _flatten

def generate_sample(x1, x2):
    """
        生成正态分布数据
        :param x1: 均值坐标
        :param x2: 均值坐标
        :return: 100个正态分布样本
        """
    N = 100
    x1 = np.random.randn(N) + x1
    x2 = np.random.randn(N) + x2
    x = np.array([x1, x2])
    x = x.T
    return x  # 100*2向量

def distEclud(x1, x2):
    """
    两个点的欧氏距离计算
    :param x1:
    :param x2:
    :return: 返回欧式距离
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def randCent(dataSet, k):
    """
    从给定数据集中随机取k个作为初始质心
    :param dataSet:
    :param k:
    :return:
    """
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  # 取0到m之间的一个整数作为下标
        centroids[i, :] = dataSet[index, :]  # 取下标为m的样本作为质心
    return centroids

def KMeans(dataSet, k):
    """
    K-Means算法实现
    :param dataSet: 数据集
    :param k: 簇的数量
    :return: centroids：质心, clusterAssment：每个样本点所属cluster
    """
    m = np.shape(dataSet)[0]  # 行的数目（样本数量）
    clusterAssment = np.mat(np.zeros((m, 2)))  # clusterAssment第一列存样本属于哪一簇，第二列存样本的到簇的中心点的误差
    clusterChange = True  # 记录质心是否改变

    # 随机初始化k个质心
    centroids = randCent(dataSet, k)
    # 运行k-Means算法
    while clusterChange:  # 循环退出条件：质心不再变化
        clusterChange = False

        # 遍历所有的样本，计算每个样本最近的质心
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心  找出最近的质心
            for j in range(k):
                distance = distEclud(centroids[j, :], dataSet[i, :])  # 计算该样本到质心的欧式距离
                if distance < minDist:  # 找出最小质心
                    minDist = distance
                    minIndex = j
            # 更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 更新新的簇的质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=_flatten(clusterAssment.astype(int)[:, 0].reshape(1,m).tolist()))  # 画图，点的颜色使用该样本点所属的类
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(0, k), marker='x')#画质心
    plt.show()


# 生成数据
dataSet = np.vstack((generate_sample(0, 0), generate_sample(3, 3), generate_sample(3, -1)))

# 画散点图
plt.scatter(dataSet[:, 0], dataSet[:, 1])
plt.show()

# 运行K-Means算法
k = 3
centroids, clusterAssment = KMeans(dataSet, k)

showCluster(dataSet, k, centroids, clusterAssment)  # 画图
