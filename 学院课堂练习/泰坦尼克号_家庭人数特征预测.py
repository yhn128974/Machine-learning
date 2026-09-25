"""
泰坦尼克号乘客遇难（死亡）预测案例 - 多特征工程对比评估
------------------------------------------------------------------
专注于对比不同特征工程方案下的分类评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率/回归率 (Recall)
"""

import os
import sys
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 避免 Windows 终端输出中文时出现乱码
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ==========================================
# 终端中英文自适应对齐格式化工具
# ==========================================
def get_display_width(text):
    """计算字符串在终端中的实际显示宽度（中文字符占2宽，英文字符占1宽）"""
    width = 0
    for ch in str(text):
        if unicodedata.east_asian_width(ch) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


def pad_str(text, width, align='left'):
    """按终端实际显示宽度对齐字符串"""
    s = str(text)
    cur_w = get_display_width(s)
    pad = max(0, width - cur_w)
    if align == 'right':
        return ' ' * pad + s
    elif align == 'center':
        left = pad // 2
        right = pad - left
        return ' ' * left + s + ' ' * right
    else:
        return s + ' ' * pad


def print_aligned_table(headers, rows, aligns=None):
    """打印完美对齐的中英文表格"""
    if aligns is None:
        aligns = ['left'] * len(headers)

    # 计算每列的最大显示宽度
    col_widths = [get_display_width(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], get_display_width(val))

    # 添加内边距
    col_widths = [w + 2 for w in col_widths]

    # 分隔线生成函数
    def make_divider(char='-'):
        parts = [char * w for w in col_widths]
        return '+' + '+'.join(parts) + '+'

    # 打印表头
    print(make_divider('-'))
    header_line = '|' + '|'.join(pad_str(h, col_widths[i], 'center') for i, h in enumerate(headers)) + '|'
    print(header_line)
    print(make_divider('='))

    # 打印数据行，每组方案之间添加分隔线
    prev_group = None
    for row in rows:
        curr_group = row[0]
        if prev_group is not None and curr_group != prev_group:
            print(make_divider('-'))
        prev_group = curr_group

        row_line = '|' + '|'.join(pad_str(val, col_widths[i], aligns[i]) for i, val in enumerate(row)) + '|'
        print(row_line)

    print(make_divider('-'))


# ==========================================
# 数据加载与处理
# ==========================================
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'titanic.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.getcwd(), 'machine learning', 'data', 'titanic.csv')
    return pd.read_csv(csv_path)


def prepare_datasets(df):
    data = df.copy()

    # 目标标签 y (1: 死亡/遇难, 0: 幸存)
    y = (data['Survived'] == 0).astype(int)

    # 缺失值填补
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # 基础类别编码
    data['Sex_Code'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked_Code'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 方案 1: 原始连续特征基准
    cols_base = ['Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Code']
    X_base = data[cols_base].copy()

    # 方案 2: 简单合成家庭人数 (连续 FamilySize 代替 SibSp / Parch)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    cols_simple = ['Pclass', 'Sex_Code', 'Age', 'FamilySize', 'Fare', 'Embarked_Code']
    X_simple = data[cols_simple].copy()

    # 方案 3: 年龄与家庭规模分桶 + One-Hot 编码
    age_bins = [0, 12, 18, 35, 60, 150]
    age_labels = ['Age_Child', 'Age_Teen', 'Age_Young', 'Age_Middle', 'Age_Senior']
    data['Age_Bin'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=True)

    def classify_family(size):
        if size == 1:
            return 'Family_Solo'
        elif 2 <= size <= 4:
            return 'Family_Small'
        else:
            return 'Family_Large'

    data['Family_Bin'] = data['FamilySize'].apply(classify_family)

    base_cols = data[['Pclass', 'Sex_Code', 'Fare', 'Embarked_Code']].copy()
    age_dummies = pd.get_dummies(data['Age_Bin'], prefix='', prefix_sep='').astype(int)
    family_dummies = pd.get_dummies(data['Family_Bin'], prefix='', prefix_sep='').astype(int)
    X_advanced = pd.concat([base_cols, age_dummies, family_dummies], axis=1)

    return [
        ('方案1: 原始连续特征 (7特征)', X_base),
        ('方案2: 简单合成家庭人数 (连续值)', X_simple),
        ('方案3: 进阶离散化分桶 (One-Hot)', X_advanced)
    ], y


def evaluate_pipeline(datasets, y):
    table_rows = []

    for name, X in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. KNN 模型
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        y_pred_knn = knn.predict(X_test_scaled)

        table_rows.append([
            f" {name}",
            "KNN (k=5)",
            f"{accuracy_score(y_test, y_pred_knn) * 100:.2f}%",
            f"{precision_score(y_test, y_pred_knn, zero_division=0) * 100:.2f}%",
            f"{recall_score(y_test, y_pred_knn, zero_division=0) * 100:.2f}%"
        ])

        # 2. 逻辑回归模型
        lr = LogisticRegression(random_state=42, max_iter=500)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        table_rows.append([
            f" {name}",
            "逻辑回归",
            f"{accuracy_score(y_test, y_pred_lr) * 100:.2f}%",
            f"{precision_score(y_test, y_pred_lr, zero_division=0) * 100:.2f}%",
            f"{recall_score(y_test, y_pred_lr, zero_division=0) * 100:.2f}%"
        ])

    return table_rows


def main():
    raw_df = load_data()
    datasets, y = prepare_datasets(raw_df)
    rows = evaluate_pipeline(datasets, y)

    headers = ["特征工程方案", "模型算法", "准确率 (Accuracy)", "精确率 (Precision)", "召回率/回归率 (Recall)"]
    aligns = ["left", "center", "center", "center", "center"]

    print("\n" + " " * 22 + "【不同特征工程方式下的模型评估指标对比】\n")
    print_aligned_table(headers, rows, aligns)
    print()


if __name__ == '__main__':
    main()
