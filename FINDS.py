import pandas as pd
from pandas import DataFrame

data=[
[1,'Sunny','Warm','Normal','Strong','Warm','Same'],
[1,'Sunny','Warm','High','Strong','Warm','Same'],
[0,'Rainy','Cold','High','Strong','Warm','Change'],
[1,'Sunny','Warm','High','Strong','Cold','Change']
]
df=DataFrame(data)

m=4 #样本个数
n=6 #特征维数
h=['null']*n

for i in range(m):
    if data[i][0] == 1:
        for j in range(1,n+1):
            if h[j-1]=='null':
                h[j-1]=data[i][j]
            elif h[j-1]!=data[i][j]:
                h[j-1]='?'
print(h)

#预测
pred=[1]*m
for i in range(m):
    for j in range(1,n+1):
        if not (h[j-1]=='?' or h[j-1]==data[i][j]):
            pred[i]=0

#计算预测准确率
cnt=0
for i in range(m):
    if pred[i]==data[i][0]:
        cnt=cnt+1

print('预测结果:',pred,'正确率：',cnt/m)