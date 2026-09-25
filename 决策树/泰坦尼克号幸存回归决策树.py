import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor  # 回归决策树
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import os


# 解决中文乱码问题 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def dm01_regression_tree():

    # 1 准备数据
    x = np.array(list(range(1,11))).reshape(-1, 1)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    print('x-->\n', x)
    print('y-->\n', y)

    # 2 实例化模型 模型训练
    model1 = DecisionTreeRegressor(max_depth=1)
    # 设置深度可以预剪枝
    model2 = DecisionTreeRegressor(max_depth=3)
    model3 = LinearRegression()
    # 模型训练
    model1.fit(x, y)
    model2.fit(x, y)
    model3.fit(x, y)

    # 3 模型预测 # 等差数组-按照间隔
    x_test = np.arange(0.0, 10.0, 0.01).reshape(-1, 1)
    y_pre1 = model1.predict(x_test)
    y_pre2 = model2.predict(x_test)
    y_pre3 = model3.predict(x_test)
    print(y_pre1.shape, y_pre2.shape, y_pre3.shape)

    # 4 结果可视化
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(x, y, label='data')

    plt.plot(x_test, y_pre1, label='max_depth=1')   # 深度1层
    plt.plot(x_test, y_pre2, label='max_depth=3')   # 深度3层
    plt.plot(x_test, y_pre3, label='linear')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('回归决策树与线性回归对比')
    # plt.legend(): 显示图例
    plt.legend()

    # 保存图片到当前脚本同目录下的 images 文件夹中（必须在 plt.show() 前调用）
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, 'line_tree.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f'图片已成功保存至: {image_path}')

    # plt.show(): 显示图片
    plt.show()


if __name__ == '__main__':
    dm01_regression_tree()


