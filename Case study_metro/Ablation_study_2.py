#Ablation study: Communities learnt by attributes

import pandas
import math
import numpy as np
from matplotlib import pyplot
import seaborn as sns
import matplotlib
from sklearn.cluster import KMeans,  AgglomerativeClustering
import csv
from scipy.stats import poisson
from scipy.stats import norm
import datetime
import scipy


###################################
#Data
Inflow_pre_weekday = np.load('Inflow_pre_weekday.npy') #(234, 91, 40)
Outflow_pre_weekday = np.load('Outflow_pre_weekday.npy') #(234, 91, 40)

L = 2
Num_month_train = 20
Inflow_train = Inflow_pre_weekday[:,:,:Num_month_train]
Inflow_test = Inflow_pre_weekday[:,:,Num_month_train:]
Outflow_train = Outflow_pre_weekday[:,:,:Num_month_train]
Outflow_test = Outflow_pre_weekday[:,:,Num_month_train:]

flow_train = np.zeros((Inflow_train.shape[0], Inflow_train.shape[1], Inflow_train.shape[2], L))
flow_train[:,:,:,0] = Inflow_train
flow_train[:,:,:,1] = Outflow_train
flow_test = np.zeros((Inflow_test.shape[0], Inflow_test.shape[1], Inflow_test.shape[2], L))
flow_test[:,:,:,0] = Inflow_test
flow_test[:,:,:,1] = Outflow_test


#basic information
st_info = np.load('st_info_pre.npy')
Num_time_pre = flow_train.shape[0]
Num_st = flow_train.shape[1]
t_T = Num_time_pre
t_1 = 0
K = st_info.shape[1] #number of attributes
L = 2 #layers
delta = 1
Num_month_train = flow_train.shape[2]
Num_month_test = flow_test.shape[2]




#Parameter initialization ########################################
R = 3 #community number


#Tal
km = KMeans(n_clusters=R)
km.fit(st_info)
y_pred1 = km.labels_
tal0 = 1/int(R-1)
Tal = (1-tal0)/(R-1) * np.ones((Num_st,R))
for i in range(0,Num_st):
    Tal[i,y_pred1[i]] = tal0



K = st_info.shape[1]
#Phi
Q = 20 #iterative number
Phi = np.ones((R,K))
Pidr = np.zeros((Num_st,R))
Q_Phi = np.ones((R,K,Q))
Q_Tal = np.zeros((Num_st,R,Q))
for q in range(0,Q):
    q
    for r in range(0,R): #community
        for i in range(0,Num_st): #station
            Pc_ir = Tal[i,r]
            #
            Px_id_cir = np.zeros((K,))
            for k in range(0,K):
                x_ik = st_info[i,k]
                Px_id_cir[k] = poisson.pmf(x_ik, Phi[r,k])   
            Px_id_cir_sum = np.prod(Px_id_cir)
            #    
            Pidr[i,r] = Pc_ir * Px_id_cir_sum 
    hat_tal_ird = Pidr / np.tile(np.reshape(Pidr.sum(axis = 1),(Num_st,1)),(1,R))
    for r in range(0,R): #community
        for k in range(0,K):
            sum_phi_0 = np.zeros((Num_st,))
            sum_phi_1 = np.zeros((Num_st,))
            for i in range(0,Num_st): #station
                sum_phi_0[i] = hat_tal_ird[i,r] * st_info[i,k]
                sum_phi_1[i] = hat_tal_ird[i,r]
            Phi[r,k] = sum_phi_0.sum() / sum_phi_1.sum()
    Q_Phi[:,:,q] = Phi
    Q_Tal[:,:,q] = hat_tal_ird
            


sns.heatmap(Phi, annot=True)
pyplot.show()

sns.heatmap(hat_tal_ird)
pyplot.show()



#community number
com = np.zeros((Num_st))
for i in range(0,Num_st): #station
    com[i] = np.argwhere(hat_tal_ird[i,:] == hat_tal_ird[i,:].max())[0,0]



myfile = open('community number_attributes.csv','w')
with myfile:
    writer = csv.writer(myfile)
    writer.writerows(np.reshape(com,(com.shape[0],1)))






#################################################################
R = 3
#Gamma is given 
Gamma = np.zeros((Num_st,Num_st,L))
for num_c in range(0,R):     
    orignal_com = pandas.read_csv('community number_attributes.csv',header=None) #station information
    com = orignal_com.values
    Num_c = np.argwhere(com==num_c)[:,0]
    for ic1 in range(0,Num_c.shape[0]):
        for ic2 in range(0,Num_c.shape[0]):
            Gamma[Num_c[ic1],Num_c[ic2],:] = 1



Mu = np.load('Initial_Mu.npy')
Alpha = np.load('Initial_Alpha.npy')


starttime = datetime.datetime.now()
#
#Parameter estimation
Q = 100 #iterative number
Num_time_pre = flow_train.shape[0]
Num_st = flow_train.shape[1]
Num_month_train = flow_train.shape[2]
L = flow_train.shape[3]
t_T = Num_time_pre
t_1 = 0
delta = 1
Q_Mu = np.zeros((Num_st,t_T-t_1,L,Q))
Q_Alpha = np.zeros((Num_st,Num_st,L,Q))
Alpha = np.ones((Num_st,Num_st,L)) * 1e-2 * Gamma
for q in range(0,Q):
    q
    for l in range(0,L):
        #l
        for i in range(0,Num_st):
            #i
            Oi = np.argwhere((Gamma[i,:,l]==1))[:,0]
            flow_train_Oi = flow_train[:,Oi,:,:]
            Num_st_Oi = Oi.shape[0]
            lambda_t = np.zeros((t_T-t_1,Num_month_train))
            Pii = np.ones((t_T-t_1,Num_month_train))
            Pij = np.zeros((t_T-t_1,Num_st_Oi,t_T-t_1,Num_month_train))
            t = 0
            lambda_t[t,:] = Mu[i,t,l]
            Sum_exp = np.zeros((Num_st_Oi, t_T-t_1,t_T-t_1,Num_month_train))
            for t in range(t_1+1, t_T):
                n_jdltj = flow_train_Oi[:t,:,:,l] 
                theta = Alpha[i,Oi,l]
                exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                sum_exp_t = np.tile(np.reshape(theta,(Num_st_Oi,1,1)),(1,t,Num_month_train)) \
                            * np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                            * np.transpose(n_jdltj,(1,0,2))
                lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
                Pii[t,:] = Mu[i,t,l] / lambda_t[t,:]
                Pij[t,:,:t,:] = sum_exp_t / np.tile(np.reshape(lambda_t[t,:],(1,1,Num_month_train)),(Num_st_Oi,t,1))
                sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                            * np.transpose(n_jdltj,(1,0,2))
                Sum_exp[:,t,:t,:] = sum_exp
            #Pii
            N_idlt = flow_train[:,i,:,l] 
            Mu[i,:,l] = (Pii * N_idlt).sum(axis = 1) / Num_month_train 
            #
            exp = np.array([np.exp( - (t_T - t_j) / delta) for t_j in range(t_1, t_T)])
            for j in range(0,Num_st_Oi):
                sum1 = (Pij[:,j,:,:] * np.tile(np.reshape(N_idlt,(t_T-t_1,1,Num_month_train)),(1,t_T-t_1,1))).sum()
                sum2 = Sum_exp[j,:,:,:].sum()
                Alpha[i,Oi[j],l] = sum1 / sum2
    Q_Mu[:,:,:,q] = Mu
    Q_Alpha[:,:,:,q] = Alpha



#
#


endtime = datetime.datetime.now()
Time2 = (endtime - starttime).seconds #2112




np.save('Mu.npy',Mu)
np.save('Alpha.npy',Alpha)




#####################################
#Model testing
Mu = np.load('Mu.npy')
Alpha = np.load('Alpha.npy')


#One-step forecast
Num_day_test = flow_test.shape[2]
hat_flow1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    #l = 0
    for i in range(0,Num_st): #station
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]
        for t in range(t_1+1, t_T):
            n_jdltj = flow_test[:t,:,:,l] 
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow1[:,i,:,l] = lambda_t
            



#Two-step forecast
hat_flow2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
hat_flow_c1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    for i in range(0,Num_st): #station
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]
        for t in range(t_1+1, t_T-1):
            n_jdltj = flow_test[:t,:,:,l] 
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow_c1[:,i,:,l] = lambda_t
        #
        lambda1_t = np.zeros((t_T-t_1,Num_month_test))
        lambda1_t[0,:] = lambda_t[0,:]
        lambda1_t[1,:] = lambda_t[1,:]
        for t in range(t_1+2, t_T):
            n_jdltj = flow_test[:t,:,:,l] * np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            n_jdltj[-1,:,:] = hat_flow_c1[t-1,:,:,l] * np.ones((hat_flow_c1[t-1,:,:,l].shape[0],hat_flow_c1[t-1,:,:,l].shape[1]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda1_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow2[:,i,:,l] = lambda1_t





#Three-step forecast
hat_flow3 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
hat_flow_c1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
hat_flow_c2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    for i in range(0,Num_st): 
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]
        for t in range(t_1+1, t_T-2):
            n_jdltj = flow_test[:t,:,:,l] 
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow_c1[:,i,:,l] = lambda_t
        #
        lambda1_t = np.zeros((t_T-t_1,Num_month_test))
        lambda1_t[0,:] = lambda_t[0,:]
        lambda1_t[1,:] = lambda_t[1,:]
        for t in range(t_1+2, t_T-1):
            n_jdltj = flow_test[:t,:,:,l] * np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            n_jdltj[-1,:,:] = hat_flow_c1[t-1,:,:,l] * np.ones((hat_flow_c1[t-1,:,:,l].shape[0],hat_flow_c1[t-1,:,:,l].shape[1]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda1_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow_c2[:,i,:,l] = lambda1_t
        #
        lambda2_t = np.zeros((t_T-t_1,Num_month_test))
        lambda2_t[0,:] = lambda_t[0,:]
        lambda2_t[1,:] = lambda_t[1,:]
        lambda2_t[2,:] = lambda1_t[2,:]
        for t in range(t_1+3, t_T):
            n_jdltj = flow_test[:t,:,:,l] * np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            n_jdltj[-2,:,:] = hat_flow_c1[t-2,:,:,l] * np.ones((hat_flow_c1[t-2,:,:,l].shape[0],hat_flow_c1[t-2,:,:,l].shape[1]))
            n_jdltj[-1,:,:] = hat_flow_c2[t-1,:,:,l] * np.ones((hat_flow_c2[t-1,:,:,l].shape[0],hat_flow_c2[t-1,:,:,l].shape[1]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda2_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow3[:,i,:,l] = lambda2_t








# Metrics: MAE, RMAE, Time; MAE and RMAE at each layer
#One-step forecast
MAE = abs(hat_flow1 - flow_test).mean() 
RMAE = abs(hat_flow1 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE =  52.873664418868636
print('RMAE = ', RMAE) #RMAE =  0.058633693978753613

#Two-step forecast
MAE = abs(hat_flow2 - flow_test).mean() 
RMAE = abs(hat_flow2 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 180.51822399181285
print('RMAE = ', RMAE) #RMAE = 0.20018378562290184

#Three-step forecast
MAE = abs(hat_flow3 - flow_test).mean() 
RMAE = abs(hat_flow3 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 275.15669511663486 
print('RMAE = ', RMAE) #RMAE =  0.30513212267384576




#For each layer
l = 0
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 49.89795801666346
print('l = 0, RMAE = ', RMAE) #RMAE = 0.05529663925052662
l = 1
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 55.84937082107384
print('l = 1, RMAE = ', RMAE) #RMAE = 0.06197523878795625


l = 0
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 187.21239610889597
print('l = 0, RMAE = ', RMAE) #RMAE = 0.2074677350003618
l = 1
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 173.82405187472972
print('l = 1, RMAE = ', RMAE) #RMAE = 0.192890035530383


l = 0
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 284.4093402343197
print('l = 0, RMAE = ', RMAE) #RMAE = 0.31518084730372053
l = 1
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 265.90404999895003
print('l = 1, RMAE = ', RMAE) #RMAE = 0.2950698772626339


       

np.save('Proposed_hat_flow1.npy',hat_flow1)
np.save('Proposed_hat_flow2.npy',hat_flow2)
np.save('Proposed_hat_flow3.npy',hat_flow3)









