from sklearn.datasets import load_iris
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# 解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 生成案例对象
iris=load_iris()

def get_data():
    # # 特征
    # print(iris.data)
    # print("--------------------------")
    # # 标签
    # print(iris.target)
    # print("--------------------------")
    df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
    # 添加标签列
    df['label']=iris.target
    print(df)
    return df


# print("--------------------------")


"""
方式二：从本地CSV文件加载
df=pd.read_csv('./data/iris.csv')
"""
def get_data2():
    df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','花的种类'])
    df['label']=iris.target
    print(df)
    return df

if __name__=='__main__':
    # df=get_data()
    # # print(df)
    iris_data=get_data2()
    #可视化查看
    # 1. matplotlib散点图
    # iris_data.plot(kind='scatter',x='花萼长度',y='花萼宽度',color='花瓣长度',colormap='viridis')
    # plt.show()
    # 2.seaborn可视化查看
    sns.set_style('whitegrid')
    # fit_reg=True表示添加回归线，fit_reg=False表示不添加回归线
    sns.scatterplot(x='花萼长度',y='花萼宽度',hue='花的种类',data=iris_data,fit_reg=True)
    plt.show()

    
