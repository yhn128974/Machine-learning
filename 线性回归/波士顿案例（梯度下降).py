# 导入相关库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
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

# 模型创建
    # 梯度下降:SGDRegressor(损失函数：squared_error(均方误差)，惩罚项：l2(岭回归)，最大迭代次数：10000，学习率：0.01，随机状态：42，学习率策略：constant(固定学习率))
    # 损失函数：squared_error(均方误差),huber(Huber损失),epsilon_insensitive(epsilon不敏感损失),squared_epsilon_insensitive(平方epsilon不敏感损失)
    # 惩罚项：l2(岭回归),l1(Lasso回归),elasticnet(弹性网络)
    # 学习率策略：constant(固定学习率),invscaling(反比例缩放),optimal(最优),adaptive(自适应)
    # 如果没有指定其他参数，则默认损失函数为squared_error(均方误差)，惩罚项为l2(岭回归)，最大迭代次数为1000，学习率策略为constant(固定学习率)，学习率为0.01，随机状态为42
    
model=SGDRegressor(loss="squared_error",penalty="l2",max_iter=10000,learning_rate="constant",eta0=0.01,random_state=42)

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

