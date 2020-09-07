import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

##loading training dataset
x_train_array = np.loadtxt(open("X_train.csv","rb"),delimiter=",",skiprows=0)
y_train_array = np.loadtxt(open("y_train.csv","rb"),delimiter=",",skiprows=0)


##loading test dataset
x_test_array = np.loadtxt(open("X_test.csv","rb"),delimiter=",",skiprows=0)
y_test_array = np.loadtxt(open("y_test.csv","rb"),delimiter=",",skiprows=0)


##preprocessing the training data
x_train = np.mat(x_train_array)
y_train = np.mat(y_train_array)
y_train = np.transpose(y_train)

##preprocessing the testing data
x_test = np.mat(x_test_array)
y_test = np.mat(y_test_array)
y_test = np.transpose(y_test)


U,S,VT = la.svd(x_train,full_matrices=False)
##此处需要对lambda循环
y_predict = []
rmse = []
L = [i for i in range(51)]
# print(L)
for l in range(0,51):
    S_element = (S/(l+S**2))
    S_lambda = np.mat(np.diag(S_element))

    V = np.mat(np.transpose(VT))
    UT = np.mat(np.transpose(U))
    ##calculate the wrr

    Wrr = V*S_lambda*UT*y_train
    y_predict = x_test*Wrr

    rmse.append((np.mean((np.array(y_test)-np.array(y_predict))**2))**0.5)

plt.title("λ-RMSE figure")
plt.plot(L,rmse)
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.legend()
plt.show()



