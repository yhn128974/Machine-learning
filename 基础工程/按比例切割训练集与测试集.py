"""
切割训练集何测集
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 导入数据
df=pd.read_csv('./machine learning/data/iris.csv',names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','花的种类'])

#划分训练集和测试集，x对应feature，y对应label
# test_size表示测试集占总数据集的比例，random_state表示随机种子，保证每次划分的结果一致
x_train, x_test, y_train, y_test = train_test_split(df[['花萼长度','花萼宽度','花瓣长度','花瓣宽度']], df['花的种类'], test_size=0.2, random_state=0)

def print_data():
    # print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    # print('---'*30)
    # # 训练集
    # print(x_train.head())
    # print('---'*30)
    # print(y_train.head())
    # print('---'*30)
    # # 测试集
    # print(x_test.head())
    # print('---'*30)
    # print(y_test.head())
    # print('---'*30)
    pass

# 标准化/归一化处理
mms=StandardScaler()
new_x_train=mms.fit_transform(x_train)
new_x_test=mms.transform(x_test)

# print(new_x_train)
# print('---'*30)
# print(new_x_test)

# 创建模型对象
knn=KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(new_x_train,y_train)

# 预测
pred=knn.predict(new_x_test)

# 评估
# 1.准确率
# 1.1 accuracy_score
# print(accuracy_score(y_test,pred))

# 1.2 knn.score
print(knn.score(new_x_test,y_test))
# print(pred==y_test)
