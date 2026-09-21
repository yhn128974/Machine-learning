import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error # 计算均方误差
from sklearn.model_selection import train_test_split


def dm01_模型欠拟合():
    # 1 准备数据x y(增加上噪声)
    np.random.seed(666)
    # 随机生成100个在-3到3之间的浮点数的数组
    x = np.random.uniform(-3, 3, size=100)
    # 真实关系y=0.5x^2+x+2+噪声
    # 生成100个随机噪声数，均值为0，方差为1
    y = np.random.normal(0, 1, size=100)+0.5 * x ** 2 + x + 2 

    # 2 实例化线性回归模型
    estimator = LinearRegression()

    # 3 训练模型
    # x.reshape(行数, 列数) 就是将一维的数据转换成符合 sklearn 规范的二维单特征矩阵，  -1表示自动计算行数，1表示只有1列
    X = x.reshape(-1, 1)
    estimator.fit(X, y)

    # 4 模型预测
    y_predict = estimator.predict(X)

    # 5 计算均方误差
    mse = mean_squared_error(y, y_predict)
    print(f"拟合函数为：{estimator.coef_[0]}*x + {estimator.intercept_}")
    print('均方误差为:', mse)

    # 6 画图
    # 原来的散点
    plt.scatter(x, y)
    # 预测的曲线，plot是画线
    plt.plot(x, y_predict, color='r')

    plt.show()

def dm02_模型ok():
    # 1 准备数据x y(增加上噪声)
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    # 这里的噪声是为了模拟真实数据中不可预测的随机波动
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2 实例化线性回归模型
    model = LinearRegression()

    # 3 训练模型
    X = x.reshape(-1, 1)
    print('X.shape-->', X.shape)
    # np.hstack（水平拼接 / horizontal stack）：将这两列按水平方向拼在一起，得到形状为 (100, 2) 的新矩阵 X2
    X2 = np.hstack([X, X ** 2]) # 数据增加二次项
    print('X2.shape-->', X2.shape)
    model.fit(X2, y)

    # 4 模型预测
    y_predict = model.predict(X2)
    print('y_predict-->', y_predict)

    # 5 计算均方误差
    mse = mean_squared_error(y, y_predict)
    print('mse-->', mse)

    # 6 画图
    plt.scatter(x, y)
    # plt.plot(x, y_predict, color=‘r’)
    # 画图plot折线图时 需要对x进行排序, 取x排序后对应的y值
    # np.argsort(x)是返回x排序后的索引值，然后根据索引值取y_predict对应的y值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show()


def dm03_模型过拟合():
    # 1 准备数据x y(增加上噪声)
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2 实例化线性回归模型
    estimator = LinearRegression()
    # 3 训练模型
    X = x.reshape(-1, 1)
    # print(‘X.shape-->’, X.shape)
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])  # 数据增加高次项
    estimator.fit(X3, y)

    # 4 模型预测
    y_predict = estimator.predict(X3)

    # 5 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret-->', myret)

    # 6 画图
    plt.scatter(x, y)
    # plt.plot(x, y_predict, color=‘r’)
    # 画图时输入的x数据: 要求是从小到大
    # np.argsort(x) 返回 x 排序后的索引值，然后根据索引值取 y_predict 对应的 y 值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show() 

if __name__ == '__main__':
   dm03_模型过拟合()