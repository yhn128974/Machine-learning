from sklearn.preprocessing import StandardScaler

train_x=[[90,2,10,40],[60,4,15,45],[75,3,13,46]]
test_x=[[100,2,8,50],[30,4,10,35]]
# 建立标准化对象
mms=StandardScaler()
# fit_transform表示既训练又转换，fit的时候会计算均值和方差，transform时会用均值和方差进行标准化
new_train_x=mms.fit_transform(train_x)
# transform表示只转换，直接使用训练集时计算得到的均值和方差进行标准化，不会重新计算均值和方差
new_test_x=mms.transform(test_x)
print(new_train_x)
print(new_test_x)

# 标准化和归一化的区别：
# 标准化：把数据转换成均值为0，方差为1的分布（减去均值除以标准差）
# 归一化：把数据转换成0到1之间的分布（减去最小值除以最大值减去最小值）

# 标准化的应用场景：
# 1.在数据量比较小的时候，可以使用标准化
# 2.在数据量比较大的时候，可以使用归一化
