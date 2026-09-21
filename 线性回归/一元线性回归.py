from sklearn.linear_model import LinearRegression

#准备数据身高和体重  
x_train = [[160], [166], [172], [174],[180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test=[[176]]

# 创建模型
model=LinearRegression()

# 模型训练
model.fit(x_train,y_train)
# 
y_predict=model.predict(x_test)
# 
print(f"测试参数为：{y_predict}")

# 了解模型的系数与截距
print('-'*30)
print(f"系数为{model.coef_},截距为{model.intercept_}")

