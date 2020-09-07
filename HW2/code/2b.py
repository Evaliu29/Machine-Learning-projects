import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math




x_array = np.loadtxt(open("X.csv","rb"),delimiter=",",skiprows=0)
print(np.shape(x_array))
y_array = np.loadtxt(open("Y.csv","rb"),delimiter=",",skiprows=0)
print("y shape",np.shape(y_array))

## calculate the pi
# print(len(y_array))


# print(sum(y_array))
# print(pi)

# new_y = 1 - y_array
# print("1-yi shape",np.shape(new_y))
# y_matrix = np.mat(y_array)
# print("y_matrix shape", np.shape(y_matrix))
# new_y_matrix = np.mat(new_y)
# x_matrix = np.mat(x_array)
#
#
# print("x matrix",np.shape(x_matrix))
# print("1-yi matrix shape",np.shape(new_y_matrix))
# ## calculate the lambda0, 1

l0 = np.zeros(54)
l1 = np.zeros(54)


col_x = np.shape(x_array)[1]

print("x数组的列数",np.shape(x_array)[1])


##cross validation
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(x_array):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_array[train_index], x_array[test_index]
    Y_train, Y_test = y_array[train_index], y_array[test_index]

    new_y = 1 - Y_train
    new_y_matrix = np.mat(new_y)
    print("1-y train shape",np.shape(new_y_matrix))
    y_matrix = np.mat(Y_train)
    print("y train matrix shape",np.shape(y_matrix))

    x_train_matrix = np.mat(X_train)
    x_test_matrix = np.mat(X_test)
    print("x train matrx shape", np.shape(x_train_matrix))
    print("x test matrx shape", np.shape(x_test_matrix))

    ## y = 1 的概率 pi_1
    ## y = 0 的概率 pi_0
    pi_1 = np.sum(Y_train) / len(Y_train)
    pi_0 = 1 - pi_1
    print("pi1",pi_1)
    print("pi0",pi_0)

    ##calculate the lambda
    lambda0d = []
    lambda1d = []
    p0 = []
    p1 = []

    for col in range(0,col_x):
        x_col = x_train_matrix[:,col]
        # print("x matrix column",np.shape(x_col))
        lambda0d.append(((1 + new_y_matrix * x_col)/(1 + np.sum(new_y)))[0,0])
        lambda1d.append(((1 + y_matrix * x_col) / (1 + np.sum(Y_train)))[0,0])


    lambda0d = np.array(lambda0d)
    lambda1d = np.array(lambda1d)
    # lambda0d = np.mat(lambda0d)
    # lambda1d = np.mat(lambda1d)

    print("lambda0d",lambda0d)
    print("l shape",np.shape(lambda0d))
    # print("lambda1d",lambda1d)

    l0 = l0 + lambda0d
    l1 = l1 + lambda1d
    # print("l0",l0)
    # print("lambda0d shape",np.shape(lambda0d))
    # print("lambda1d shape",np.shape(lambda1d))
    # print("lambda0d type",type(lambda0d))
    # print("lambda1d type",type(lambda1d))

    # print(np.log(lambda0d))
    # print(np.shape(np.log(lambda0d)))
    X_test_row = len(X_test)
    for row in range(X_test_row):
        # print(X_test[row,:])
        # print(np.shape(X_test[row,:]))
        p0_tmp1 = np.multiply(X_test[row,:],(np.log(lambda0d)))
        # print(p0_tmp1)
        p0_tmp2 = p0_tmp1 - lambda0d
        # print(p0_tmp2)
        p0_sum = np.sum(p0_tmp2)
        # print("sum",p0_sum)
        pi0_ln = np.log(pi_0)
        p0.append(pi0_ln + p0_sum)

        #
        ##p = 1
        # print(np.log(lambda1d))
        # print(np.shape(np.log(lambda1d)))
        # print(X_test[row,:])
        # print(np.shape(X_test[row,:]))
        p1_tmp1 = np.multiply(X_test[row,:],(np.log(lambda1d)))
        # print(p1_tmp1)
        p1_tmp2 = p1_tmp1 - lambda1d
        # print(p1_tmp2)
        p1_sum = np.sum(p1_tmp2)
        # print("sum",p1_sum)
        pi1_ln = np.log(pi_1)
        p1.append(pi1_ln + p1_sum)
    # print("p1",p1)
    # print("p0", p0)
    # print("p1 shape",np.shape(p1))
    # print("p0 shape",np.shape(p0))
    p = np.greater(p1,p0)
    y_pred = p.astype(int)
    # print(y_pred)

l0 = l0/10
l1 = l1/10
print("l0",l0)
print("l1",l1)
x = [i for i in range(0,54)]
plt.title('54 Poisson parameters-class diagram')
plt.xlabel('dimension')
plt.ylabel('Poisson parameters')
plt.stem(x,l0,linefmt='b',markerfmt='o', label='non-spam email')
plt.stem(x,l1, linefmt='r',markerfmt='^', label = 'spam email')
plt.legend()
plt.show()

#
# fig = plt.figure("lambda parameters")
# axes0 = plt.subplot(211)
# axes0.set_title('Lambda for y=0')
# axes0.set_xlabel('d')
# axes0.set_ylabel('Lambda0-value')
# axes0.stem(x,l0,linefmt='b',markerfmt='o')
#
#
# axes1 = plt.subplot(212)
# axes1.set_title('Lambda for y=1')
# axes1.set_xlabel('d')
# axes1.set_ylabel('Lambda1-value')
# axes1.stem(x,l1, linefmt='r',markerfmt='o')
# plt.show()
#
