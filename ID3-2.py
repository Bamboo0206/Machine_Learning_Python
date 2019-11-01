'''
递归划分
    递归返回：数据集内全是一个标签的
    选取信息增益最大的特征
        对每一个特征，求信息增益。选出最大的返回。
    将数据集划分为k个子集，对每个子集递归划分
'''


import math

def CreateDataSet():
    dataset=[[0,0,0,0,0],
             [0,0,0,1,0],
             [0,1,0,1,1],
             [0,1,1,0,1],
             [0,0,0,0,0],
             [1,0,0,0,0],
             [1,0,0,1,0],
             [1,1,1,1,1],
             [1,0,1,2,1],
             [1,0,1,2,1],
             [2,0,1,2,1],
             [2,0,1,1,1],
             [2,1,0,1,1],
             [2,1,0,2,1],
             [2,0,0,0,0]]
    feature_name=['age','have job','have house','credit']

    return dataset, feature_name

def createTree(dataSet,feature_name):
    label = data[:, -1]  # 取最后一列标签
    if label.count(label[0])==len(label):#如果所有的训练数据都是属于一个类别，则返回该类别
        return label[0]
#创建树
def createTree(dataSet, feature_name):
    classList = dataSet[:, -1]    #创建需要创建树的训练数据的结果列表（例如最外层的列表是[N, N, Y, Y, Y, N, Y]）
    if classList.count(classList[0]) == len(classList):  #如果所有的训练数据都是属于一个类别，则返回该类别
        return classList[0];
    if (len(dataSet[0]) == 1):  #训练数据只给出类别数据（没给任何特征值数据），返回出现次数最多的分类名称
        return majorityCnt(classList);

    bestFeat = FindMaxInfoGainFeature(dataSet);   #选择信息增益最大的特征进行分（返回值是特征类型列表的下标）
    bestFeatLabel = feature_name[bestFeat]  #根据下表找特征名称当树的根节点 #找名称
    myTree = {bestFeatLabel:{}}  #以bestFeatLabel为根节点建一个空树
    del(feature_name[bestFeat])  #从特征列表中删掉已经被选出来当根节点的特征
    featValues = [example[bestFeat] for example in dataSet]  #找出该特征所有训练数据的值（创建列表）
    uniqueVals = set(featValues)  #求出该特征的所有值得集合（集合的元素不能重复）
    for value in uniqueVals:  #根据该特征的值求树的各个分支
        subfeature_name = feature_name[:]
        myTree[bestFeatLabel][value] = createTree(FiltrateDataSet(dataSet, bestFeat, value), subfeature_name)  #根据各个分支递归创建树
    return myTree  #生成的树


data=CreateDataSet()
# label=data[:,-1]#取最后一列标签