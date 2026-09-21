
from sklearn.neighbors import KNeighborsRegressor

train_x=[[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
train_y=[0,0,1,1]
# 
test_x=[[4,5,6]] 

# 建立最近邻回归模型
model=KNeighborsRegressor(n_neighbors=3)

model.fit(train_x,train_y)

test_y=model.predict(test_x)

print(f'预测数据为{test_y}')




