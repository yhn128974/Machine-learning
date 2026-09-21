# 解决二分问题
# 导包
from sklearn.linear_model import LogisticRegression 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import os

# 数据准备
# 获取当前脚本所在目录的上层目录中的 data 文件夹，确保无论在哪个目录下运行都能正确找到文件
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'breast-cancer-wisconsin.csv')
data = pd.read_csv(data_path)
# print(data.head())

# 数据预处理（缺失值处理，异常值处理）
data=data.replace(to_replace='?',value=np.nan)
# 删除数据集中所有包含缺失值（NaN / 空值）的整行样本
data.dropna(inplace=True)
# 1:-1 表示从第1列到倒数第二列的所有列，0表示第一列（ID列），所以删除ID列和最后一列（Target列）
features=data.iloc[:,1:-1]
# -1 表示最后一列（Target列）
target=data.iloc[:,-1]
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
# 准确率：预测正确的样本数 / 总样本数 （通过testx预测testy的结果，然后进行比较，看预测对了几个）
print("测试集上的准确率：{:.2%}".format(score))
# 混淆矩阵：TP  FP
#           FN  TN
# TP：真正例，实际为正，预测为正
# FP：假正例，实际为负，预测为正
# FN：假反例，实际为正，预测为负
# TN：真反例，实际为负，预测为负





