# 导包
from sklearn.linear_model import LogisticRegression 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report


# 准备数据:
# 有10个样本，6个恶性肿瘤，4个良性肿瘤，我们假设恶性为正样本，良性为负样本
test_y = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

# 模拟模型A的预测结果：预测对了3个恶性肿瘤样本，4个良性肿瘤样本
# (3个恶性判为恶性，3个恶性误判为良性；4个良性判为良性)
predict_a = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']

# 模拟模型B的预测结果：预测对了6个恶性肿瘤样本，1个良性肿瘤样本
# (6个恶性判为恶性；3个良性误判为恶性，1个良性判为良性)
predict_b = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性']

# 创建混淆矩阵
# 参数一：真实结果，参数二：预测结果，参数三：标签名labels=[正例, 反例]
# 注意：normalize参数的可选值为 None, 'true', 'pred', 'all'，不能传 False
cm_a = confusion_matrix(test_y, predict_a, labels=['恶性', '良性'])
print(f"混淆矩阵A:\n{cm_a}")
cm_b = confusion_matrix(test_y, predict_b, labels=['恶性', '良性'])
print(f"混淆矩阵B:\n{cm_b}")

# 使用pd.DataFrame显示混淆矩阵
df_cm_a=pd.DataFrame(cm_a,index=['真实悪性','真实良性'],columns=['预测悪性','预测良性'])
df_cm_b=pd.DataFrame(cm_b,index=['真实悪性','真实良性'],columns=['预测悪性','预测良性'])
print(f"混淆矩阵A:\n{df_cm_a}")
print(f"混淆矩阵B:\n{df_cm_b}")
print(f"\n")

# 准确率：预测正确的样本数/总样本数
print(f"模型a的准确率为：{accuracy_score(test_y,predict_a)}")
print(f"模型b的准确率为：{accuracy_score(test_y,predict_b)}")
print(f"\n")

# 精确率：预测正确的正样本数/预测为正样本的总数
print(f"模型a的精确率为：{precision_score(test_y,predict_a,pos_label='恶性')}")
print(f"模型b的精确率为：{precision_score(test_y,predict_b,pos_label='恶性')}")
print(f"\n")

# 召回率：预测正确的正样本数/实际为正样本的总数
print(f"模型a的召回率为：{recall_score(test_y,predict_a,pos_label='恶性')}")
print(f"模型b的召回率为：{recall_score(test_y,predict_b,pos_label='恶性')}")
print(f"\n")

# F1值：精确率和召回率的调和平均值
print(f"模型a的F1值为：{f1_score(test_y,predict_a,pos_label='恶性')}")
print(f"模型b的F1值为：{f1_score(test_y,predict_b,pos_label='恶性')}")
print(f"\n")

# TODO:模型A和模型B生成roc_auc分数以及分类评估报告，注意这里的标签必须是数值类型


# 标签映射
label_map = {'恶性': 1, '良性': 0}
test_y_numeric = [label_map[x] for x in test_y]
predict_a_numeric = [label_map[x] for x in predict_a]
predict_b_numeric = [label_map[x] for x in predict_b]

# 计算 roc_auc 分数
auc_a = roc_auc_score(test_y_numeric, predict_a_numeric, )
auc_b = roc_auc_score(test_y_numeric, predict_b_numeric, )
print(f"模型A的 roc_auc 分数：{auc_a}")
print(f"模型B的 roc_auc 分数：{auc_b}")

print(f"\n")

# 计算分类评估报告
report_a = classification_report(test_y, predict_a)
report_b = classification_report(test_y, predict_b)
print("模型A的分类评估报告：\n", report_a)
print("模型B的分类评估报告：\n", report_b)
print(f"\n")

# 数据预处理

# 特征工程

# 模型创建

# 模型训练

# 模型预测

# 模型评估（混淆矩阵）