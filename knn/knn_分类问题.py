"""
k-近邻
# 1.准备数据
# 2.计算距离
# 3.按照距离排序，找到最相似的k个样本
# 4.表决找出个数最多的类别
# 5.预测结果  
"""

from sklearn.neighbors import KNeighborsClassifier

train_x=[[0],[1],[2],[3]]
train_y=[0,0,1,1]
test_x=[[4]]

# # 建立最近邻分类模型，设置最近邻个数
model=KNeighborsClassifier(n_neighbors=3)

model.fit(train_x,train_y)  

test_y=model.predict(test_x)

print(f"预测结果为{test_y}") 




