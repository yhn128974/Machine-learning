import numpy as np
list1=np.array([[1,2,3],[4,5,6]])
list2=np.array([[4,5,6],[7,8,9]])

# 矩阵点乘的写法
print(list1.dot(list2.T))
print(np.dot(list1,list2.T))
print(list1@list2.T)
