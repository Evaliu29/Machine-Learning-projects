import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math

def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))




x_array = np.loadtxt(open("X.csv","rb"),delimiter=",",skiprows=0)
print(np.shape(x_array))
y_array = np.loadtxt(open("Y.csv","rb"),delimiter=",",skiprows=0)
print("y shape",np.shape(y_array))

K = [i for i in range(1,21)]


#
# x_array = np.loadtxt(open("X.csv","rb"),delimiter=",",skiprows=0)
# print(np.shape(x_array))
# y_array = np.loadtxt(open("Y.csv","rb"),delimiter=",",skiprows=0)
# print("y shape",np.shape(y_array))

## calculate the l1 distance
def get_l1_distance(train, test):
    differ = np.abs(train - test)
    l1 = np.sum(differ)
    return l1


l0 = np.zeros(54)
l1 = np.zeros(54)


col_x = np.shape(x_array)[1]

print("x数组的列数",np.shape(x_array)[1])



acc_list = []
##cross validation
kf = KFold(n_splits=10, shuffle=True)
for k in range(1,21):
    print("k",k)
    acc_num = 0
    for train_index, test_index in kf.split(x_array):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_array[train_index], x_array[test_index]
        Y_train, Y_test = y_array[train_index], y_array[test_index]
        # print(len(X_train))
        ##先做一个测试点的,之后循环每一个test case
        y_pre = []

        for row in range(len(X_test)):

            x_test = X_test[row,:]

            # print("x_test",x_test)
            # print(np.shape(x_test))
            # print("x_train",X_train)
            # print(np.shape(X_train))
            l1 = np.apply_along_axis(get_l1_distance,1,X_train,x_test)
            # print("l1",l1)
            # print(np.shape(l1))
            ##最近的1个目前
            index = np.argsort(l1)[:k]
            # print(l1[index])
            # print(index)
            res = Y_train[index]
            # print("res", res)
            res_sum = np.sum(res)
            if res_sum >= len(res)/2:
                ans = 1
            else:
                ans = 0
            # print(ans)
            y_pre.append(ans)
        y_pre = np.array((y_pre))
        # print("preres",y_pre)
        # print("Y_test",Y_test)
        # print("y_pre",len(y_pre))

        TP = find_TP(Y_test, y_pre)
        TN = find_TN(Y_test, y_pre)
        acc_num += TP + TN
        # print("acc_num",acc_num)


    acc = acc_num / 4600
    acc_list.append(acc)
print("total acc", acc_list)


plt.plot(K, acc_list)
plt.xlabel('K')
plt.ylabel('Prediction accuracy')
plt.title('K-Prediction accuracy')
plt.show()








