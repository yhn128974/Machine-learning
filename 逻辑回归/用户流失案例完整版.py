# 导包
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report

# 确保中文字体显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 导入数据
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'churn.csv')
data = pd.read_csv(data_path)
# print(data.head())
# print(data.info())

# 数据预处理 
# TODO:pd.get_dummies(): 独热编把二分类型转化为两列（转化为数值类型）
# drop_first: 独热编码后的数据,删除第一列. 这样可以避免多重共线性.
#  如果设置为True, 会删除第一列. 如果设置为False, 不会删除第一列.
data = pd.get_dummies(data,drop_first=True)
# print(data.head())
# TODO:修改列名 data.rename(columns={'旧列名':'新列名'},inplace=True)
data.rename(columns={'Churn_Yes':'flag'},inplace=True)
data.rename(columns={'gender_Female':'gender'},inplace=True)
# print(data.head())
# print('-'*30)
# print(data['flag'].value_counts())

# # 展示数据
# print('-'*30)
# print(data.columns)

# # 可视化展示信息  
# print('-'*30)
# # 设置图片大小
# plt.figure(figsize=(6,4))
# sns.countplot(x='Contract_Month',hue='flag',palette='rainbow',data=data)
# plt.show()

# 1:-1 表示从第1列到倒数第二列的所有列，0表示第一列（ID列），所以删除ID列和最后一列（Target列）
features=data[['Contract_Month', 'PaymentElectronic', 'internet_other']]
# -1 表示最后一列（Target列）
target=data['flag']
train_features,test_features,train_target,test_target=train_test_split(features,target,test_size=0.2,random_state=10)

# 特征工程（归一化，标准化）
# 标准化
# StandardScaler() 拟合训练数据并转换训练数据，然后仅对测试数据应用相同的转换
scaler=StandardScaler()
train_features=scaler.fit_transform(train_features)
test_features=scaler.transform(test_features)

# 模型创建（根据业务选择模型）
model=LogisticRegression()

# 模型训练
model.fit(train_features,train_target)
# 模型预测
test_predict=model.predict(test_features)
# 模型评估
score=accuracy_score(test_target,test_predict)
# TODO:1.准确率：预测正确的样本数 / 总样本数 （通过testx预测testy的结果，然后进行比较，看预测对了几个）
print("测试集上的准确率：{:.2%}".format(score))

# TODO: 2.精确率
# TP / (TP + FP)
# 模型预测为正类中, 实际为正类的样本数 / 模型预测为正类的总数
precision = precision_score(test_target, test_predict)
print(f"精确率: {precision:.2%}")

# TODO: 3.召回率
# TP / (TP + FN)
recall = recall_score(test_target, test_predict)
print(f"召回率: {recall:.2%}")

# TODO: 4.F1值
f1 = f1_score(test_target, test_predict)
print(f"F1值: {f1:.2%}")



