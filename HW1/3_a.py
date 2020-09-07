import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


x_train_array = np.loadtxt(open("X_train.csv","rb"),delimiter=",",skiprows=0)
# print(type(x_train))
y_train_array = np.loadtxt(open("y_train.csv","rb"),delimiter=",",skiprows=0)


x_train = np.mat(x_train_array)
y_train = np.mat(y_train_array)
y_train = np.transpose(y_train)

# print(np.shape(y_train))
# print(np.shape(x_train))
U,S,VT = la.svd(x_train,full_matrices=False)
##此处需要对lambda循环
d1,d2,d3,d4,d5,d6,d7 = [], [] ,[], [], [], [], []
df_lamba = []
for l in range(0,5001):
    S_element = (S/(l+S**2))
    S_lambda = np.mat(np.diag(S_element))

    V = np.mat(np.transpose(VT))
    UT = np.mat(np.transpose(U))
    ##calculate the wrr
    Wrr = V*S_lambda*UT*y_train
    Wrr_arr = np.array(Wrr)

    d1.append(Wrr_arr[0])
    d2.append(Wrr_arr[1])
    d3.append(Wrr_arr[2])
    d4.append(Wrr_arr[3])
    d5.append(Wrr_arr[4])
    d6.append(Wrr_arr[5])
    d7.append(Wrr_arr[6])

    S_d = S**2/(l+S**2)
    df = np.sum(S_d)
    df_lamba.append(df)

plt.plot(df_lamba,d1,label = "cylinders")
plt.plot(df_lamba,d2,label = "displacement")
plt.plot(df_lamba,d3,label = "horsepower")
plt.plot(df_lamba,d4,label = "weight")
plt.plot(df_lamba,d5,label = "acceleration")
plt.plot(df_lamba,d6,label = "year made")
plt.plot(df_lamba,d7)

plt.title("df(λ)-Wrr figure")
plt.xlabel('df(λ)')
plt.ylabel('Wrr')
plt.legend()
plt.show()







