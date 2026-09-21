"""
KNN 手写数字图像识别案例
目标：使用 KNN 算法，结合当前学过的知识，识别 data 目录下的 demo.png 图片中的手写数字。
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------
# 1. 确定文件路径（适配当前脚本所在目录）
# -------------------------------------------------------------
# 获取当前脚本所在目录的绝对路径，确保在任何地方运行都能准确找到平行目录下的 data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'data', '手写数字识别.csv')
img_path = os.path.join(base_dir, 'data', 'demo.png')

print("正在加载手写数字训练集...")
# 手写数字识别.csv 包含 785 列：
# 第 0 列是标签 'label'（数字 0~9）
# 第 1 到 784 列是 28*28 像素点的灰度值（pixel0 ~ pixel783）
# 提示：为保证快速演示运行，此处可以加载前 5000 行（也可去除 nrows 参数加载全量数据）
df = pd.read_csv(csv_path, nrows=5000)

# -------------------------------------------------------------
# 2. 提取特征 (X) 和标签 (y)
# -------------------------------------------------------------
# 此时第一列是标签，后面所有列是特征：
# 提取特征：从第 1 列开始取到最后一列（二维 DataFrame，共 784 列）
x_train = df.iloc[:, 1:]
# 提取标签：取第 0 列（一维 Series，即真实数字）
y_train = df.iloc[:, 0]

print(f"训练集特征维度: {x_train.shape}, 标签数量: {y_train.shape[0]}")

# -------------------------------------------------------------
# 3. 读取并处理待识别图片 (demo.png)
# -------------------------------------------------------------
# 打开图片并转为灰度图（单通道）
img = Image.open(img_path).convert('L')

# 将 28*28 的二维像素矩阵转换为 numpy 数组，并展平成 1 行 784 列的一维向量
# 必须和训练集特征维度（784列）保持一致，且满足输入模型所需的二维结构 (1, 784)
img_arr = np.array(img).reshape(1, -1)

print(f"待识别图片转换后的特征维度: {img_arr.shape}")

# -------------------------------------------------------------
# 4. 特征工程：归一化处理（防止极端特征/像素值差异对欧氏距离造成过大影响）
# -------------------------------------------------------------
# 得到归一化对象，将像素特征缩放到 [0, 1] 区间
mms = MinMaxScaler(feature_range=(0, 1))

# 对训练集进行归一化（fit计算特征min/max，transform进行缩放）
new_x_train = mms.fit_transform(x_train.values)

# 对待预测图片进行归一化（复用训练集的min/max标准，调用transform）
new_img_arr = mms.transform(img_arr)

# -------------------------------------------------------------
# 5. 创建 KNN 模型并训练
# -------------------------------------------------------------
# 实例化 KNN 模型，这里选择近邻数 k=3
knn = KNeighborsClassifier(n_neighbors=3)

# 拟合归一化后的训练数据
knn.fit(new_x_train, y_train.values)

# -------------------------------------------------------------
# 6. 预测 demo.png 的数字
# -------------------------------------------------------------
prediction = knn.predict(new_img_arr)

print("---" * 20)
print(f"识别完成！demo.png 中的手写数字预测结果为: {prediction[0]}")
print("---" * 20)
