import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math


####------problem 1------####

##generate 3 gaussians###
cov = np.array([[1,0],[0,1]])
mean1 = np.array([0,0])
mean2 = np.array([3,0])
mean3 = np.array([0,3])
# print(cov.shape)
gaussian1 = np.random.multivariate_normal(mean1, cov,500)
gaussian2 = np.random.multivariate_normal(mean2, cov,500)
gaussian3 = np.random.multivariate_normal(mean3, cov,500)
# print(gaussian1.shape)

##generate 500 data points###
choice = np.random.choice(a=3, size=500, p=[0.2, 0.5, 0.3])
# print(gau_choice)
data1 = gaussian1[choice == 0,:]
data2 = gaussian2[choice == 1,:]
data3 = gaussian3[choice == 2,:]

data = np.concatenate((data1,data2,data3))
print("data",data.shape)


# print(data.shape)
# k = 2
###---part a---####

def get_closet_distance(row,u):
    # print("row shape",row.shape)
    # print("u",u.shape)
    # print(((row-u)**2).shape)
    dist = np.sum(((row - u)**2), axis = 1)
    print("dist shape",dist.shape)
    # print("dist",dist)
    min_dist_index = np.argmin(dist)
    min_dist = dist[min_dist_index]

    # print("min_dist",min_dist)
    # print("min_dist shape", min_dist.shape)
    return min_dist_index,min_dist
K = [2,3,4,5]
cluster3 = []
cluster5 = []
assign_cluster = []
for k in K:
    loss = []
    ##random initialize u
    u = np.random.uniform(low=0, high=1, size=(k, 2))
    ##iteration 20
    for iter in range(20):
        assign_cluster = np.apply_along_axis(get_closet_distance,1,data,u)
        # print(assign_cluster)
        # print(assign_cluster.shape)
        # print(type(assign_cluster))
        # print("assign cluster", assign_cluster[:,1])
        loss.append(np.sum(assign_cluster[:,1]))



        ##update uk
        for i in range(k):
            u[i,:] = np.mean(data[assign_cluster[:,0]==i], axis = 0)
            # print(u[i].shape)

        # print(u.shape)
        # print("iter u",u)

    print("k",k)
    print(loss)
    plt.plot(range(1,21),loss)
    if k == 3:
        cluster3 = assign_cluster[:,0]
        # print(cluster3)
    if k == 5:
        cluster5 = assign_cluster[:,0]

clusters = ["blue", "red","green","purple","yellow"]
c3 = [clusters[int(i)] for i in cluster3]
# print(c3)
c5 = [clusters[int(i)] for i in cluster5]
plt.figure(1)
plt.xticks(range(1,21))
plt.xlabel('20 iterations')
plt.ylabel('Objective function value')
plt.title('Objective function per iteration for K = [2,3,4,5]')
plt.legend(['K = %d'%i for i in K])

plt.figure(2)
plt.title('Cluster diagram for K = 3')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.scatter(data[:,0], data[:,1],c = c3)
plt.figure(3)
plt.title('Cluster diagram for K = 5')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.scatter(data[:,0], data[:,1],c = c5)
plt.show()








