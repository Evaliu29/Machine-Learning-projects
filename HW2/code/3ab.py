import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd

##loading the training data
x_train_array = np.loadtxt(open("X_train.csv","rb"),delimiter=",",skiprows=0)
print("x train",np.shape(x_train_array))
y_train_array = np.loadtxt(open("y_train.csv","rb"),delimiter=",",skiprows=0)
print("y_train", np.shape(y_train_array))

##loading test dataset
x_test_array = np.loadtxt(open("X_test.csv","rb"),delimiter=",",skiprows=0)
print("x test",np.shape(x_test_array))
y_test_array = np.loadtxt(open("y_test.csv","rb"),delimiter=",",skiprows=0)
print("y_test", np.shape(y_test_array))



sigma2 = 0.1
b = 5

def calculate_K(x1,x2,b,row1,row2):
    K = [[0 for m in range(row2)] for m in range(row1)]
    for i in range(row1):
        for j in range(row2):
            dis = np.dot(x1[i]-x2[j],x1[i]-x2[j])
            K[i][j] = np.exp(-1 * dis / b)
    # print("Kernel",np.shape(K))
    # K = np.array(K)
    return K

def find_mean(Kr,sigma2,Kn,y_train_array):
    n = np.shape(Kn)[0]
    I = np.identity(n)
    inv = np.linalg.inv((sigma2 * I + Kn))
    tmp = np.dot(Kr,inv)
    mean = np.dot(tmp,y_train_array)
    # print(mean)
    return mean


def find_product_cov(Kr,sigma2,Kn):
    n = np.shape(Kn)[0]
    I = np.identity(n)
    inv = np.linalg.inv((sigma2 * I + Kn))
    # print("inv",np.shape(inv))
    # print(np.shape(Kr))
    # product = Kr.dot(inv).dot(Kr)
    product = np.dot(np.dot(Kr,inv),np.transpose(Kr))
    # print(type(product))
    # print("pro",np.shape(product))
    return product

def cal_mean (x_test_array,x_train_array,b,sigma2,y_train_array):
    mean = []
    Kn = calculate_K(x_train_array, x_train_array, b, np.shape(x_train_array)[0], np.shape(x_train_array)[0])

    for i in range(np.shape(x_test_array)[0]):
        Kxd = calculate_K_4(x_test_array[i, :], x_train_array, b, 1, np.shape(x_train_array)[0])
        # print(np.shape(Kxd))
        mean.append(find_mean(Kxd, sigma2, Kn, y_train_array)[0])
    return mean




def cal_mean_4 (x_test_array,x_train_array,b,sigma2,y_train_array):
    mean = []
    Kn = calculate_K(x_train_array, x_train_array, b, np.shape(x_train_array)[0], np.shape(x_train_array)[0])

    for i in range(np.shape(x_test_array)[0]):
        Kxd = calculate_K_4(x_test_array[i], x_train_array, b, 1, np.shape(x_train_array)[0])
        # print(np.shape(Kxd))
        # Kxx = calculate_K(x_test_array[i], x_test_array[i], b, 1, 1)
        mean.append(find_mean(Kxd, sigma2, Kn, y_train_array)[0])
    return mean

def calculate_K_4(x1,x2,b,row1,row2):
    K = [[0 for m in range(row2)] for m in range(row1)]
    for i in range(row1):
        for j in range(row2):
            dis = np.dot(x1-x2[j],x1-x2[j])
            K[i][j] = np.exp(-1 * dis / b)
    # print("Kernel",np.shape(K))
    # K = np.array(K)
    return K


def calculate_K_1(x1,x2,b,row1,row2):
    K = [[0 for m in range(row2)] for m in range(row1)]
    for i in range(row1):
        for j in range(row2):
            dis = np.dot(x1-x2,x1-x2)
            K[i][j] = np.exp(-1 * dis / b)
    # print("Kernel",np.shape(K))
    # K = np.array(K)
    return K



def cal_cov(x_train_array,x_test_array,b,sigma2):
    covariance = []
    Kn = calculate_K(x_train_array, x_train_array, b, np.shape(x_train_array)[0], np.shape(x_train_array)[0])
    for i in range(np.shape(x_test_array)[0]):
        Kxd = calculate_K_4(x_test_array[i], x_train_array, b, 1, np.shape(x_train_array)[0])
        # print(np.shape(Kxd))
        Kxx = calculate_K_1(x_test_array[i, :], x_test_array[i, :], b, 1, 1)
        # print("Kxx",Kxx)
        # Kxx =calculate_K(x_test_array[i,:],x_test_array[i,:],b,1,1)
        # pro = find_product_cov(Kxd,sigma2,Kn)[0][0]

        n = np.shape(Kn)[0]
        I = np.identity(n)
        inv = np.linalg.inv((sigma2 * I + Kn))
        # print("inv",np.shape(inv))
        # print(np.shape(Kr))
        # product = Kr.dot(inv).dot(Kr)
        pro = np.dot(np.dot(Kxd, inv), np.transpose(Kxd))

        # print("pro",pro)
        Kxx = Kxx[0][0]
        # print(pro)
        # print(Kxx)
        cov = sigma2 + Kxx - pro
        # print("cov",cov)
        covariance.append(cov)
    # print("Kxx",Kxx)
    return covariance


m = cal_mean(x_test_array,x_train_array,b,sigma2,y_train_array)
covariance = cal_cov(x_train_array,x_test_array,b,sigma2)
print("mean",m)
print("cov",covariance)
# print(np.shape(mean))
# print(np.shape(covariance))

# #
b = [5,7,9,11,13,15]
sigma2 = [round(i,1) for i in np.arange(0.1,1.1,0.1)]
#

# b = 5
# sigma2 = 2
def part2b(x_train_array,x_test_array,b,sigma2,y_test_array,y_train_array):
    b_rmse = []

    for i in b:
        Kn = calculate_K(x_train_array, x_train_array, i, np.shape(x_train_array)[0], np.shape(x_train_array)[0])
        sig_rmse = []
        for j in sigma2:
            m = cal_mean(x_test_array,x_train_array,i,j,y_train_array)
            y_pre = m
            sig_rmse.append((np.mean((np.array(y_test_array) - np.array(y_pre)) ** 2)) ** 0.5)
        b_rmse.append(sig_rmse)

    print(b_rmse)
    print(np.shape(b_rmse))
    return b_rmse
# b_rmse = part2b(x_train_array,x_test_array,b,sigma2,y_test_array,y_train_array)
# res = pd.DataFrame(index=b,columns=sigma2,data=b_rmse)
# print(res)
# res.to_csv('3b_result.csv')

# print(x_train_array[:,3][0])


