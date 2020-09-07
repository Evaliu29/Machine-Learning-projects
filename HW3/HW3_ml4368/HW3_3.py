import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
import pandas as pd
import sys

####------problem 3------####
####parta####
##### sigma2 = 0.25, d = 10 and lambda = 1

N1 = 943##number of users
N2 = 1682##number of movies
d = 10##rank
lam = 1
sigma2 = 0.25
run_num = 10
iteration = 100

def pre_error(M):
    observed_index = ~np.isnan(M)
    predict = U.dot(V)
    # print(predict.shape)
    error = ((M[observed_index] - predict[observed_index])**2).sum()
    # print(error)
    return error

def dist_sqare(x):
    return (x**2).sum()

##initialize M
M = np.repeat(np.nan,N1*N2).reshape(N1,N2)
# print(type(m))
# print(m.shape)
# print(m)
M_data = np.loadtxt(open("Prob3_ratings.csv","rb"),delimiter=",",skiprows=0)
# print(m_data.shape)
# print(m_data)
for r in M_data:
    # print(r[0])
    M[int(r[0])-1,int(r[1])-1] = float(r[2])

# print(m)
##predict on the test
pred_test = []
pred_true = []
total_rmse = []
final_joint = []
final_V = np.repeat(0,d*N2).reshape(d,N2)
max = - sys.float_info.max

for run in range(run_num):
    total_joint = []
    ##initialize ui,vj
    U = np.repeat(np.nan, N1 * d).reshape(N1, d)
    # print(U.shape)
    # print(type(U))
    cov = np.identity(d)/lam
    # print(cov)
    mean = np.zeros(d)
    gaussian = np.random.multivariate_normal(mean, cov, N2)
    # print(gaussian.shape)
    V = gaussian.T
    # print(V.shape)

    for iter in range(iteration):
        for i in range(N1):
            # print(type(ui))
            # print(ui)
            # print(M[i])
            observed_index = np.argwhere((~np.isnan(M[i,:])))##oberved_index for each user
            # print(observed_index)
            observed_index = observed_index.flatten()
            # print(observed_index)
            vj = V[:,observed_index]
            # print(M[i,observed_index])
            # print(vj.shape)
            Mij = M[i,observed_index].reshape((1,-1))
            # print(M[i,observed_index].reshape((1,-1)).shape)
            # print(vj.T.shape)
            # print(Mij.dot((vj.T)).T.shape)
            # print(vj.dot(vj.T).shape)
            U[i,:] = np.linalg.inv(lam*sigma2*np.identity(d) + vj.dot(vj.T)).dot(Mij.dot((vj.T)).T).T
            # print(U[i,:].shape)
            # break
        # print(U.shape)
        # print(U)

        for j in range(N2):
            observed_index = np.argwhere((~np.isnan(M[:,j])))  ##oberved_index for each user
            observed_index = observed_index.flatten()
            # print(observed_index)
            # print(observed_index.shape)

            ui = U[observed_index,:].T
            # print(ui.shape)
            Mij = M[observed_index,j]
            # print("M",Mij.shape)
            # print(ui.dot(Mij).shape)
            V[:,j] = np.linalg.inv(lam*sigma2*np.identity(d) + ui.dot(ui.T)).dot(ui.dot(Mij))
            # print(V[:,j].shape)
            # print(V[:,j])
        # print(V)
        #     print(V.shape)
        if iter > 0:
            joint_likelihood = - (1 / (2 * sigma2)) * pre_error(M) - (lam / 2) * dist_sqare(U) - (lam / 2) * dist_sqare(V)
            total_joint.append(joint_likelihood)

    x = list(range(2,101))
    plt.plot(x,total_joint,label='run_%d'%(run+1))

    final_joint.append(total_joint[-1])

    ##find max V
    if total_joint[-1] > max:
        max = total_joint[-1]
        final_V = V
        print("max run",run)


###----RMSE---####
    # M_test = np.repeat(0,N1*N2).reshape(N1,N2)
    M_testdata = np.loadtxt(open("Prob3_ratings_test.csv","rb"),delimiter=",",skiprows=0)
    # print(M_data.shape)
    # print(M_data)
    for r in M_testdata:
        # print(r[0]-1)

        ui = U[int(r[0]-1),:]
        # print(ui)
        # print(ui.shape)
        vj = V[:,int(r[1]-1)]
        # print(vj.shape)
        pred_test.append(ui.dot(vj))
        pred_true.append(r[2])

    # print(np.array(pred_test).shape)
    # print(np.array(pred_true).shape)

    rmse =(np.mean(((np.array(pred_true) - np.array(pred_test)) ** 2)) ** 0.5)
    total_rmse.append(rmse)
    print("run",run+1)
    # print("rmse",rmse)

print("joint",final_joint)
print("rmse",total_rmse)
print("final V",final_V.shape)

run_list = np.array([i+1 for i in range(run_num)]).reshape(run_num,1)
sort_joint_ind = np.argsort(-np.array(final_joint))
final_joint_tmp = np.array(final_joint)[sort_joint_ind].reshape(len(final_joint),1)

total_rmse_tmp = np.array(total_rmse)[sort_joint_ind].reshape(len(total_rmse),1)
tmp = np.append(run_list,final_joint_tmp,axis = 1)
ans = np.append(tmp,total_rmse_tmp,axis = 1)




col_name = ["rank","value of training objective function","RMSE"]
res =pd.DataFrame(columns=col_name,data=ans)
print(res)
res.to_csv('result.csv',index = False)


plt.xticks([int(x) for x in np.linspace(2,100,10)])
plt.xlabel('100-Iterations')
plt.ylabel(' Log joint likelihood')
plt.title(' Log joint likelihood for 10 runs')
plt.legend()
plt.show()

###---partb---####
file = "Prob3_movies.txt"
with open(file) as f:
    movies = np.array([x.rstrip('\n') for x in f.readlines()])
# print(movies)##1682*1
index = np.argwhere(movies == 'Star Wars (1977)')[0][0]
print(index)
index1 = np.argwhere(movies == 'My Fair Lady (1964)')[0][0]
print(index1)
index2 = np.argwhere(movies == 'GoodFellas (1990)')[0][0]
print(index2)


dist_SW = []
dist_MFL = []
dist_GF = []

for i in range(N2):
    dist_SW.append(np.sum(((final_V[:,i] - final_V[:,index])**2))**0.5)
    dist_MFL.append(np.sum(((final_V[:, i] - final_V[:, index1]) ** 2)) ** 0.5)
    dist_GF.append(np.sum(((final_V[:, i] - final_V[:, index2]) ** 2)) ** 0.5)

sort_index_SW = np.argsort(dist_SW)
print("SW 10 close index",sort_index_SW[1:11])

sort_index_SW = sort_index_SW[1:11].tolist()
# print(type(sort_index_SW))
dist_SW = np.sort(dist_SW)
# print(type(dist))
# print(dist)
print("SW 10 dists",dist_SW[1:11])
print("SW 10 movies",movies[sort_index_SW])



sort_index_SW = np.array(sort_index_SW).reshape(len(sort_index_SW),1)
dist_SW_reshape = np.array(dist_SW[1:11]).reshape(len(dist_SW[1:11]),1)
m = np.array(movies[sort_index_SW]).reshape(len(movies[sort_index_SW]),1)
tmp1 = np.append(sort_index_SW,m,axis = 1)
ans1 = np.append(tmp1,dist_SW_reshape,axis = 1)

print("Star war")

col_name = ["Movie Id","Movie Name","Euclidean distance"]
res1 =pd.DataFrame(columns=col_name,data=ans1)
print(res1)

res1.to_csv('result_SW.csv',index = False)



sort_index_MFL = np.argsort(dist_MFL)
print("MFL 10 close index",sort_index_MFL[1:11])

sort_index_MFL = sort_index_MFL[1:11].tolist()
# print(type(sort_index_SW))
dist_MFL = np.sort(dist_MFL)
# print(type(dist))
# print(dist)
print("MFL 10 dists",dist_MFL[1:11])
print("MFL 10 movies",movies[sort_index_MFL])



sort_index_MFL = np.array(sort_index_MFL).reshape(len(sort_index_MFL),1)
dist_MFL_reshape = np.array(dist_MFL[1:11]).reshape(len(dist_MFL[1:11]),1)
m2 = np.array(movies[sort_index_MFL]).reshape(len(movies[sort_index_MFL]),1)
tmp2 = np.append(sort_index_MFL,m2,axis = 1)
ans2 = np.append(tmp2,dist_MFL_reshape,axis = 1)

print("MFL")

col_name = ["Movie Id","Movie Name","Euclidean distance"]
res2 =pd.DataFrame(columns=col_name,data=ans2)
print(res2)

res2.to_csv('result_MFL.csv',index = False)




sort_index_GF = np.argsort(dist_GF)
print("GF 10 close index",sort_index_GF[1:11])

sort_index_GF = sort_index_GF[1:11].tolist()
# print(type(sort_index_SW))
dist_GF = np.sort(dist_GF)
# print(type(dist))
# print(dist)
print("GF 10 dists",dist_GF[1:11])
print("GF 10 movies",movies[sort_index_GF])




sort_index_GF = np.array(sort_index_GF).reshape(len(sort_index_GF),1)
dist_GF_reshape = np.array(dist_GF[1:11]).reshape(len(dist_GF[1:11]),1)
m3 = np.array(movies[sort_index_GF]).reshape(len(movies[sort_index_GF]),1)
tmp3 = np.append(sort_index_GF,m3,axis = 1)
ans3 = np.append(tmp3,dist_GF_reshape,axis = 1)

print("GF")

col_name = ["Movie Id","Movie Name","Euclidean distance"]
res3 =pd.DataFrame(columns=col_name,data=ans3)
print(res3)

res3.to_csv('result_GF.csv',index = False)



# print(movies[])

