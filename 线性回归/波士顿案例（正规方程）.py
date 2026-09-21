# 导入相关库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
# 加载数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r'\s+', skiprows=22, header=None)
# raw_df.values: array[ 行开始 : 行结束 : 行步长 , 列开始 : 列结束 : 列步长 ]
data=np.hstack([raw_df.values[::2,:],  raw_df.values[1::2,:2]])
target=raw_df.values[1::2,2]

# 数据集切割
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 数据预处理（标准化）
standardScaler = StandardScaler()
x_train = standardScaler.fit_transform(x_train)
x_test = standardScaler.transform(x_test)

# 特征选择

# 模型创建(正规方程或者梯度下降)

model=LinearRegression()

# 模型训练
model.fit(x_train, y_train)

# 模型预测
print(f"模型系数{model.coef_}")
print(f"模型截距{model.intercept_}")
y_predict = model.predict(x_test)
print(f"预测结果为{y_predict}")

# 模型评估
print(f"模型评估{model.score(x_test,y_test)}")  
print(f"均方误差{mean_squared_error(y_test,y_predict)}")
print(f"均方根误差{root_mean_squared_error(y_test,y_predict)}")
print(f"平均绝对误差{mean_absolute_error(y_test,y_predict)}")

