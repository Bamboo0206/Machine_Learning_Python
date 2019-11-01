import numpy as np
import matplotlib.pyplot as plt


def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    """
    M1 = np.mean(samples, axis=0)  # 求该类所有样本均值
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))  # 申请一个n*n的0矩阵 (n为特征数）
    for s in samples:  # 计算协方差矩阵
        t = s - M1
        cov_m += t * t.reshape(2, 1)
    return cov_m, M1


def fisher(positive, negative):
    """
     fisher算法实现。
    :param positive: 正样本
    :param negative: 负样本
    :return: 参数ω
    """
    cov_1, M1 = cal_cov_and_avg(positive)  # 计算协方差矩阵和平均向量
    cov_2, M2 = cal_cov_and_avg(negative)  # 计算协方差矩阵和平均向量
    s_w = cov_1 + cov_2  # 类内离散度
    u, s, v = np.linalg.svd(s_w)  # 奇异值分解（考虑到数值解的稳定性，求Sw的逆矩阵，一般先对Sw奇异值分解再求逆
    s_w_inv = np.dot(np.dot(v, np.linalg.inv(np.diag(s))), u.T)  # 求Sw的逆矩阵
    omega = np.dot(s_w_inv, M1 - M2)  # 参数ω
    return omega


def judge(X_test, y_test, w, p_train, n_train):
    """
    计算测试集预测正确率
    :param X_test: 测试集
    :param y_test: 测试集label
    :param w: 参数ω
    :param p_train: 训练集-正样本
    :param n_train: 训练集-负样本
    :return: 返回测试集正确分类的概率
    """
    u1 = np.mean(positive, axis=0)  # 计算样本质心
    u2 = np.mean(negative, axis=0)
    m1 = np.dot(w.T, u1)  # 计算质心在轴上的投影
    m2 = np.dot(w.T, u2)

    cnt = X_test.shape[0]
    pred_right = 0
    for i in range(cnt):
        pos = np.dot(w.T, X_test[i].T)  # 若测试样本在轴上的投影
        if (((abs(pos - m1) <= abs(pos - m2)) and y_test[i])  # 预测和实际都为正样本
                or ((abs(pos - m1) >= abs(pos - m2)) and (not y_test[i]))):  # 预测和实际都为负样本
            pred_right += 1
    return pred_right / cnt


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


# 生成数据
positive = generate_sample(0, 0)  # 100*2向量
negative = generate_sample(2, 5)
plt.scatter(positive[:, 0], positive[:, 1])
plt.scatter(negative[:, :1], negative[:, 1:], c="red")
plt.show()

# 将数据分为训练集和测试集
p_train = np.array(positive[:70, :])
n_train = np.array(negative[:70, :])
X_test = np.vstack([positive[70:, :], negative[70:, :]])
y_test = np.array([1] * 30 + [0] * 30)

w = fisher(p_train, n_train)  # 调用函数，得到参数w

rate = judge(X_test, y_test, w, p_train, n_train)  # 计算测试集预测正确率
print("测试集预测正确率：", rate)

# 画图
plt.scatter(p_train[:, 0], p_train[:, 1])
plt.scatter(n_train[:, 0], n_train[:, 1], c='red')

k = w[1] / w[0]  # 计算投影轴斜率
x_list = [-3, 8]  # 计算两个直线上的点：取两个x（这两个点的取值需要根据正负样本的质心相应改动）
y_list = [k * x_list[0], k * x_list[1]]  # 计算x对应坐标y
plt.plot(x_list, y_list, 'k-')  # 画投影轴
plt.show()
