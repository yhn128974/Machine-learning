import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge,LinearRegression
from sklearn.metrics import mean_squared_error


def dm04_模型过拟合_L1正则化():
    # 1 准备数据 x, y (增加噪声)
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)


    # estimator=LinearRegression()

    # 2 实例化 L1 正则化模型
    # 做实验: alpha 惩罚力度越来越大，k 值越来越小，过大会欠拟合
    # 注: 在 scikit-learn 高版本中若 normalize 参数报错，可去掉或使用 StandardScaler 预处理
    try:
        estimator = Lasso(alpha=0.005, normalize=True)
    except TypeError:
        estimator = Lasso(alpha=0.005)

    # 3 训练模型
    X = x.reshape(-1, 1)
    # np.hstack: 水平拼接数组，就是按列拼接 （扩展特征矩阵）
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 数据增加高次项
    estimator.fit(X3, y)
    # estimator.coef_:
    print('estimator.coef_:', estimator.coef_)

    # 4 模型预测
    y_predict = estimator.predict(X3)

    # 5 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret-->', myret)

    # 6 画图
    plt.scatter(x, y)
    # 画图时输入的 x 数据要求是从小到大
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.title("L1 Regularization (Lasso)")
    plt.show()


def dm05_模型过拟合_L2正则化():
    # 1 准备数据 x, y (增加噪声)
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)




    # 2 实例化 L2 正则化模型 (Ridge 回归)
    # alpha 是L2正则化的超参数，控制正则化的强度（惩罚系数）
    try:
        estimator = Ridge(alpha=0.05, normalize=True)
    except TypeError:
        estimator = Ridge(alpha=0.05)

    # 3 训练模型
    X = x.reshape(-1, 1)
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    estimator.fit(X3, y)
    print('estimator.coef_:', estimator.coef_)

    # 4 模型预测
    y_predict = estimator.predict(X3)

    # 5 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret-->', myret)

    # 6 画图
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.title("L2 Regularization (Ridge)")
    plt.show()


if __name__ == '__main__':
    # dm04_模型过拟合_L1正则化()
    dm05_模型过拟合_L2正则化()
