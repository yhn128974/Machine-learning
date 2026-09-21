"""

根据身高预测体重，利用回归方程，将一个或者多个自变量跟因变量进行分析的过程
    分类：
         一元线性回归：只有一个自变量 y = ax + b
         多元线性回归：有多个自变量 y = ax + by + cz + d

    线性回归对象：LinearRegression()
    
    评估线性回归模型：损失函数
        实现损失函数的方法：
            1.正规方程法
            2.梯度下降法
        损失函数有以下几种：
            误差：y_test - y_predict
            平均绝对误差（MAE）： 误差绝对值的和/样本数
            均方误差（MSE）： 误差平方和/样本数
            根均方误差（RMSE）： 均方误差的平方根 
            决定系数（R^2）： 1 - (误差平方和/总平方和)

    机器学习的建模流程:
        1.数据准备
        2.数据预处理
        3.特征工程
        4.模型训练
        5.模型预测
        6.模型评估
"""

from sklearn.linear_model import LinearRegression
# 1.数据准备
train_x=[[160],[166],[172],[178],[184]]

train_y=[56.3,60.6,65.1,68.9,72.6]

test_x=[[169],[174]]


# 2.数据预处理

# 3.特征工程

# 4.模型训练

# 4.1 创建模型对象
model=LinearRegression()

# 4.2 训练模型
model.fit(train_x,train_y)

# 4.3 输出回归方程
print(f"a的值为：{model.coef_[0]},b的值为：{model.intercept_}")

# 5.模型预测
print(f"169预测值为：{model.predict([[169]])}")
print(f"174预测值为：{model.predict([[174]])}")

# 6.模型评估

