"""
泰坦尼克号乘客遇难（死亡）预测案例
--------------------------------------------
教学重点：
1. 数据加载与特征理解
2. 缺失值处理与分类特征编码
3. 训练集与测试集划分（train_test_split）
4. 数据标准化处理（StandardScaler: 训练集 fit_transform，测试集 transform）
5. 模型训练与预测（KNN分类器 / 逻辑回归）
6. 模型评估（准确率、混淆矩阵、分类报告）
7. 实际样本推理预测
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 避免 Windows 终端输出中文时出现乱码
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')



# ==========================================
# 1. 准备与加载数据
# ==========================================
def load_data():
    # 动态获取数据路径，适配不同的执行环境
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'titanic.csv')
    
    # 备用路径兼容
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.getcwd(), 'machine learning', 'data', 'titanic.csv')
    
    print(f"[INFO] 正在从以下路径读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] 数据读取成功，样本总数: {df.shape[0]}, 特征总数: {df.shape[1]}")
    return df


# ==========================================
# 2. 数据清洗与特征工程
# ==========================================
def preprocess_data(df):
    """
    清洗数据并提取核心特征：
    - 选取有效特征：Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    - 剔除无因果特征：PassengerId, Name, Ticket, Cabin
    - 处理缺失值与类别编码
    """
    data = df.copy()

    # 1) 提取预测目标标签：预测死亡
    # 原始数据中 Survived: 1=生还, 0=遇难/死亡
    # 此处定义目标 y: 1 表示死亡/遇难, 0 表示幸存
    y = (data['Survived'] == 0).astype(int)

    # 2) 挑选特征列
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[feature_cols].copy()

    # 3) 缺失值填补
    # 年龄 Age 使用中位数填补
    X['Age'] = X['Age'].fillna(X['Age'].median())
    # 票价 Fare 使用中位数填补
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    # 登船港口 Embarked 使用众数填补
    mode_embarked = X['Embarked'].mode()[0]
    X['Embarked'] = X['Embarked'].fillna(mode_embarked)

    # 4) 分类特征数值编码
    # 性别：male -> 0, female -> 1
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    # 登船港口：S -> 0, C -> 1, Q -> 2
    X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return X, y


# ==========================================
# 3. 主流程：划分、标准化、训练、评估与预测
# ==========================================
def main():
    # 1. 加载数据
    raw_df = load_data()

    # 2. 预处理
    X, y = preprocess_data(raw_df)
    print("\n[INFO] 处理后的前 5 行特征数据：")
    print(X.head())
    print("\n[INFO] 死亡标签分布 (1:遇难/死亡, 0:幸存):")
    print(y.value_counts())

    # 3. 划分训练集与测试集 (遵循Featurization Ordering，切分后再进行标准化拟合)
    # test_size=0.2 表示 20% 为测试集，random_state=42 保证结果可复现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] 训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    # 4. 数据标准化处理 (StandardScaler)
    # 注意：
    # 1) 训练集使用 fit_transform()，同时计算均值和标准差并完成转换
    # 2) 测试集必须使用 transform()，直接复用训练集的均值和标准差，防止数据泄露
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[INFO] 标准化前训练集特征均值与方差举例 (Age, Fare):")
    print(f"  原始 Age: 均值={X_train['Age'].mean():.2f}, 方差={X_train['Age'].var():.2f}")
    print(f"  原始 Fare: 均值={X_train['Fare'].mean():.2f}, 方差={X_train['Fare'].var():.2f}")
    print("[INFO] 标准化后训练集特征均值与方差 (应接近 0 和 1):")
    print(f"  处理后 Age: 均值={X_train_scaled[:, 2].mean():.2f}, 方差={X_train_scaled[:, 2].var():.2f}")
    print(f"  处理后 Fare: 均值={X_train_scaled[:, 5].mean():.2f}, 方差={X_train_scaled[:, 5].var():.2f}")

    # 5. 模型训练 (以KNN模型为主，对比逻辑回归)
    # 5.1 KNN 分类器
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    # 5.2 逻辑回归模型
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    # 6. 模型评估与预测
    y_pred_knn = knn_model.predict(X_test_scaled)
    y_pred_lr = lr_model.predict(X_test_scaled)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    print("\n" + "=" * 50)
    print("               模型评估结果")
    print("=" * 50)
    print(f"KNN 模型预测准确率:      {acc_knn * 100:.2f}%")
    print(f"逻辑回归模型预测准确率:  {acc_lr * 100:.2f}%")

    print("\n[KNN 混淆矩阵] (行: 真实值, 列: 预测值):")
    print(confusion_matrix(y_test, y_pred_knn))

    print("\n[KNN 分类报告] (类别 1 为死亡/遇难, 类别 0 为幸存):")
    print(classification_report(y_test, y_pred_knn, target_names=['幸存(0)', '死亡(1)']))

    # 7. 案例推理：预测新乘客是否死亡
    print("=" * 50)
    print("        根据新乘客特征进行遇难(死亡)预测")
    print("=" * 50)

    # 模拟构造 2 位乘客的特征数据：
    # 特征顺序: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # 乘客 A: 1等舱、女性(1)、25岁、无亲属(0,0)、票价150元、登船C(1)
    # 乘客 B: 3等舱、男性(0)、30岁、无亲属(0,0)、票价7.8元、登船S(0)
    sample_passengers = pd.DataFrame([
        [1, 1, 25.0, 0, 0, 150.0, 1],
        [3, 0, 30.0, 0, 0, 7.8, 0]
    ], columns=X.columns)

    # 必须使用同一个 scaler 进行转换！
    sample_scaled = scaler.transform(sample_passengers)

    # 使用模型预测
    sample_preds = knn_model.predict(sample_scaled)
    sample_probs = knn_model.predict_proba(sample_scaled)

    passenger_names = ["乘客A (一等舱女性)", "乘客B (三等舱男性)"]
    for i, name in enumerate(passenger_names):
        pred_label = sample_preds[i]
        death_prob = sample_probs[i][1] * 100
        status = "【预测死亡/遇难】" if pred_label == 1 else "【预测幸存】"
        print(f"{name}: 死亡概率 = {death_prob:.1f}% -> 预测结果: {status}")


if __name__ == '__main__':
    main()
