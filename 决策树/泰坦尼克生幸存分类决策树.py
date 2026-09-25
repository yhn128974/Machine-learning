import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
import os

def dm04_titanic():
    # 1 读数据到内存
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'titanic.csv')
    data = pd.read_csv(data_path)
    # data.head()        # 查看前5条数据
    # data.info()         # 查看特性信息

    # 2 数据基本处理
    # 2-1 确定x y
    x = data[['Pclass', 'Age', 'Sex']]
    y = data['Survived']

    # 2-2 缺失值处理
    x['Age'].fillna(x['Age'].mean(), inplace=True)

    # 2-3 pclass类别型数据,需要转数值one-hot编码
    # print('x-->1\n', x)
    # x.info()
    # pd.get_dummies(): 独热编把二分类型转化为两列（转化为数值类型）
    # drop_first: 独热编码后的数据,删除第一列. 这样可以避免多重共线性. 
    # 如果设置为True, 会删除第一列. 如果设置为False, 不会删除第一列.
    x = pd.get_dummies(x)
    # print('x-->2\n', x)
    
    x.info()
    # 2-4 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=33)

    # 归一化
    mms=MinMaxScaler()
    x_train=mms.fit_transform(x_train)
    x_test=mms.transform(x_test)

    # 3 实例化决策树模型 训练模型
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # 4 模型预测
    y_pred = model.predict(x_test)

    # 5 模型评估
    # 5-1 输出预测准确率
    myret = model.score(x_test, y_test)
    print('myret-->\n', myret)

    # 5-2 更加详细的分类性能
    myreport = classification_report(y_pred, y_test, target_names=['died', 'survived'])
    print('myreport-->\n', myreport)

    # 6 决策树可视化
    plt.figure(figsize=(30, 20))
    plot_tree(model,
              max_depth=10,
              filled=True,
              feature_names=['Pclass', 'Age', 'Sex_female', 'Sex_male'],
              class_names=['died', 'survived'] )
    plt.show()

if __name__ == '__main__':
    dm04_titanic()  