from sklearn.preprocessing import MinMaxScaler
# 1.准备数据
train_x=[[90,2,10,40],[60,4,15,45],[75,3,13,46]]
test_x=[[100,2,8,50],[30,4,10,35]]
# 2.数据预处理
# 3.特征工程
    # 3.1得到归一化对象
mms=MinMaxScaler(feature_range=(0,1))

    # 3.2对训练集进行归一化
        # 训练集用 fit_transform(train_x)：
        # fit：从训练数据中计算出每个特征的 min 和 max
        # transform：用这些 min/max 把训练数据归一化到 [0, 1]
new_train_x=mms.fit_transform(train_x)

    # 3.3对测试集进行归一化
        # 直接复用训练集 fit 时学到的 min/max 来转换测试数据
        # 不重新计算测试集自己的 min/max
new_test_x=mms.transform(test_x)
print(new_train_x)


# 4.模型训练
# 5.模型预测
# 6.模型评估
# 7.模型上线