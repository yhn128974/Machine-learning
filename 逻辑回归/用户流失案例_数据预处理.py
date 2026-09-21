# 导包
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
print(data.head())
print('-'*30)
print(data['flag'].value_counts())

# 展示数据
print('-'*30)
print(data.columns)

# 可视化展示信息  
print('-'*30)
sns.countplot(data['Contract_Month'],hue='flag')
plt.show()

# 特征工程

# 模型初始化
# 模型训练
# 模型预测
# 模型评估
