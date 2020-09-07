import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


def find_rmse(x_train_array,y_train_array,x_test_array,y_test_array):
    x_train = np.mat(x_train_array)
    y_train = np.mat(y_train_array)
    y_train = np.transpose(y_train)

    ##preprocessing the testing data
    x_test = np.mat(x_test_array)
    y_test = np.mat(y_test_array)
    y_test = np.transpose(y_test)

    U, S, VT = la.svd(x_train, full_matrices=False)
    ##此处需要对lambda循环
    y_predict = []
    rmse = []
    for l in range(0, 101):
        S_element = (S / (l + S ** 2))
        S_lambda = np.mat(np.diag(S_element))

        V = np.mat(np.transpose(VT))
        UT = np.mat(np.transpose(U))
        ##calculate the wrr
        Wrr = V * S_lambda * UT * y_train
        y_predict = x_test * Wrr

        rmse.append((np.mean((np.array(y_test) - np.array(y_predict)) ** 2)) ** 0.5)
    return rmse




##loading training dataset
## p = 1
x_train_array_1 = np.loadtxt(open("X_train.csv","rb"),delimiter=",",skiprows=0)

##p = 2,3
x_train_array = np.loadtxt(open("X_train.csv","rb"),delimiter=",",skiprows=0)[:,:6]

y_train_array = np.loadtxt(open("y_train.csv","rb"),delimiter=",",skiprows=0)


##loading test dataset
##p = 1
x_test_array_1 = np.loadtxt(open("X_test.csv","rb"),delimiter=",",skiprows=0)
##p =2,3
x_test_array = np.loadtxt(open("X_test.csv","rb"),delimiter=",",skiprows=0)[:,:6]

y_test_array = np.loadtxt(open("y_test.csv","rb"),delimiter=",",skiprows=0)


t1 = find_rmse(x_train_array_1,y_train_array,x_test_array_1,y_test_array)

##p = 2
# print(x_train_array)
for i in range(0,6):
    a = (x_train_array[:, i] ** 2).reshape((-1, 1))
    a_test = (x_test_array[:, i] ** 2).reshape((-1, 1))
    # c = (a[:,i]**3).reshape((-1,1))
    a_mean = np.mean(a)
    a_test_mean = np.mean(a_test)
    a_n = (a-a_mean) /((np.mean((a-a_mean)**2))**0.5)##std()
    a_test_n = (a_test - a_test_mean) / ((np.mean((a_test - a_test_mean) ** 2)) ** 0.5)  ##std()
    x_train_array = np.hstack((x_train_array, a_n))
    x_test_array = np.hstack((x_test_array, a_test_n))
# print(np.shape(x_train_array))
#
# print(np.shape(x_test_array))

t2 = find_rmse(x_train_array,y_train_array,x_test_array,y_test_array)

##p = 3
for i in range(0,6):
    a = (x_train_array[:, i] ** 3).reshape((-1, 1))
    a_test = (x_test_array[:, i] ** 3).reshape((-1, 1))
    a_mean = np.mean(a)
    a_test_mean = np.mean(a_test)
    a_n = (a-a_mean) /((np.mean((a-a_mean)**2))**0.5)##std()
    a_test_n = (a_test - a_test_mean) / ((np.mean((a_test - a_test_mean) ** 2)) ** 0.5)  ##std()
    x_train_array = np.hstack((x_train_array, a_n))
    x_test_array = np.hstack((x_test_array, a_test_n))

t3 = find_rmse(x_train_array,y_train_array,x_test_array,y_test_array)

L = [i for i in range(101)]
plt.title("λ-RMSE figure")
plt.plot(L,t1,label = "p = 1")
plt.plot(L,t2,label = "p = 2")
plt.plot(L,t3,label = "p = 3")
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.legend()
plt.show()


