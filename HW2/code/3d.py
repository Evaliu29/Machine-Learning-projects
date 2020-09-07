import numpy as np
import matplotlib.pyplot as plt

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

sigma2 = 2
b = 5


def calculate_K(x1,x2,b):

    row1 = np.shape(x1)[0]
    print("row1",row1)

    row2 = np.shape(x2)[0]
    print("row2", row2)
    K = [[0 for i in range(row2)] for i in range(row1)]
    for i in range(row1):
        for j in range(row2):
            dis = np.dot(x1[i]-x2[j],x1[i]-x2[j])
            K[i][j] = np.exp(-1/b*dis)
    print("Kernel",np.shape(K))
    K = np.array(K)
    return K

# calculate_K(x_train_array[0,:],x_train_array,5)
print(type(np.array([x_train_array[:,3]])))
print(np.shape(x_train_array[:,3]))

Kn = calculate_K(x_train_array[:,3],x_train_array[:,3],b)
# print(type(Kn))
Kxd =calculate_K(x_test_array[:,3],x_train_array[:,3],b)
# print(np.shape(Kxd))
Kxx =calculate_K(x_test_array[:,3],x_test_array[:,3],b)




def find_mean(Kr,sigma2,Kn,y_train_array):
    n = np.shape(Kn)[0]
    I = np.identity(n)

    inv = np.linalg.inv((sigma2 * I + Kn))
    mean = np.dot(np.dot(Kr,inv),y_train_array)
    # print(mean)
    return mean

##all test case mean
mean = np.apply_along_axis(find_mean,1,Kxd,sigma2,Kn,y_train_array)
print(mean)
print(np.shape(mean))
print(type(mean))

#scatter figure
plt.title('Scatter Plot')
plt.xlabel('x[4]-car weight')
plt.ylabel('y-miles per gallon')
index = np.argsort(x_test_array[:,3])

plt.scatter(x_train_array[:,3],y_train_array,c = 'b',marker = 'o')
plt.plot(x_test_array[:,3][index],mean[index], linewidth = '2',c = 'r')
plt.legend()
plt.show()

