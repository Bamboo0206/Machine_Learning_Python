import math


def H(dataset):  # 计算随机变量的熵
    sampleCnt = len(dataset)  # 样本总个数
    labelCounts = {}
    for featVec in dataset:  # 统计每个类别的的样本个数，存储于dict里（key=label，value=统计个数）
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:  # 计算熵
        shannonEnt -= float(labelCounts[key]) / sampleCnt * math.log(float(labelCounts[key]) / sampleCnt, 2)
    return shannonEnt


def CreateDataSet():
    dataset = [[0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 1, 0, 1, 1],
               [0, 1, 1, 0, 1],
               [0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [1, 0, 0, 1, 0],
               [1, 1, 1, 1, 1],
               [1, 0, 1, 2, 1],
               [1, 0, 1, 2, 1],
               [2, 0, 1, 2, 1],
               [2, 0, 1, 1, 1],
               [2, 1, 0, 1, 1],
               [2, 1, 0, 2, 1],
               [2, 0, 0, 0, 0]]
    feature_name = ['age', 'have job', 'have house', 'credit']
    return dataset, feature_name


# 筛选出数据集中第FeatIx个特征 取值等于value的样本，删除第FeatIx列特征，并组成一个list并返回
def FiltrateDataSet(dataSet, FeatIx, value):
    retDataSet = []
    for featVec in dataSet:  # 按dataSet矩阵中的第FeatIx列的值等于value的分数据集
        if featVec[FeatIx] == value:  # 值等于value的，每一行为新的列表（去除第FeatIx个数据）
            reducedFeatVec = featVec[:FeatIx]  # 删除第FeatIx列特征
            reducedFeatVec.extend(featVec[FeatIx + 1:])  # 删除第FeatIx列特征
            retDataSet.append(reducedFeatVec)
    return retDataSet


def FindMaxInfoGainFeature(dataSet):  # 选取信息增益最大的特征
    numberFeatures = len(dataSet[0]) - 1  # 特征数（列数-1，最后一列为label
    Entropy = H(dataSet)
    MaxInfoGain = 0.0
    bestFeatIndexIndex = -1
    for i in range(numberFeatures):  # 对每一个特征，计算信息增益
        featList = set([example[i] for example in dataSet])  # 获取该特征所有可能取值
        ConditionalEntropy = 0.0  # 条件熵
        for value in featList:  # 计算第i个特征的条件熵
            subDataSet = FiltrateDataSet(dataSet, i, value)  # 过滤出第i个特征取值等于value的子数据集
            ConditionalEntropy += (len(subDataSet) / float(len(dataSet))) * H(subDataSet)  # 求i列特征各值对于的熵求和
        infoGain = Entropy - ConditionalEntropy  # 求出第i列特征的信息增益
        if (infoGain > MaxInfoGain):  # 保存信息增益最大的增量以及所在的下表（列值i）
            MaxInfoGain = infoGain
            bestFeatIndexIndex = i
    return bestFeatIndexIndex


def CreateDecisionTree(dataSet, feature_name):  # 建立决策树
    label = [example[-1] for example in dataSet]  # 取最后一列标签
    if label.count(label[0]) == len(label):  # 如果所有的训练数据都是属于一个类别，则返回该类别
        return label[0]

    bestFeatIndex = FindMaxInfoGainFeature(dataSet)  # 选择信息增益最大的特征进行分（返回值是特征类型列表的下标）
    bestFeatIndexName = feature_name[bestFeatIndex]  # 找到该特征名称
    myTree = {bestFeatIndexName: {}}  # 以bestFeatIndexName为根节点建一个空树
    del (feature_name[bestFeatIndex])  # 从特征列表中删掉已经被选出来当根节点的特征

    featValues = set([example[bestFeatIndex] for example in dataSet])  # 找出该特征所有可能取值
    for value in featValues:  # 将数据集划分为k个子集，对每个子集递归划分
        subfeature_name = feature_name[:]
        myTree[bestFeatIndexName][value] = CreateDecisionTree(FiltrateDataSet(dataSet, bestFeatIndex, value),
                                                              subfeature_name)
    return myTree


Data, feature_name = CreateDataSet()
Tree = CreateDecisionTree(Data, feature_name)
# print(CreateDecisionTree(Data, feature_name))
print(Tree)

# 测试算法

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点名称
    secondDict = inputTree[firstStr]  # 节点对应出的分支
    featIndex = featLabels.index(firstStr)  # 跟节点对应的属性名称
    classLabel = None
    for key in secondDict.keys():  # 对每个分支循环
        if testVec[featIndex] == key:  # 测试样本进入某个分支
            if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:  # 如果是叶子， 返回结果
                classLabel = secondDict[key]
    return classLabel


# Tree = CreateDecisionTree(Data, feature_name)
feature_name = ['age', 'have job', 'have house', 'credit']
cnt=0
for testData in Data:
    dic = classify(Tree, feature_name, testData)
    if testData[-1]==dic:
        cnt=cnt+1
    print(dic)

print(cnt)