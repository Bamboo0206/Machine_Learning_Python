import math
import pandas
from pandas import DataFrame

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


df=DataFrame()

#求熵：1. 按正负样本将数据集划分为两部分
#2.求log
#熵
def H(sub_df,category): #参数要传入dataFrame和分类依据（字符串）
    cnt=sub_df.groupby(category).count()
    return -p*math.log(p,2)-(1-p)*math.log((1-p),2)

# def g(D,A):
#     return h

n=5 #特征维数
while True: #建立树
    maxG=0
    maxi=-1
    for i in range(n): #求每个特征的信息增益
       m= df.count() #样本总数
       g=H()
       for D in df.groupby[i]: #求第i个特征的信息增益
           groupCnt=D.i.count()
           Di=D[D['loan']==1].count()/groupCnt
           g=g-(groupCnt/m)*H(Di)
        if maxG<g: #找到信息增益最大的特征
            maxG=g
            maxi=i
    print(maxi) #找到了最大
    #判断Di是否还需要继续划分