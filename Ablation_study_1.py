#Ablation study: Neither attributes nor communities

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



#################################################################
#Parameter initialization
R = 3
Mu = np.load('Initial_Mu.npy')
Alpha = np.load('Initial_Alpha.npy')




starttime = datetime.datetime.now()
#
#Parameter estimation
Q = 400 #iterative number
Num_time_pre = flow_train.shape[0]
Num_st = flow_train.shape[1]
Num_month_train = flow_train.shape[2]
L = flow_train.shape[3]
t_T = Num_time_pre
t_1 = 0
delta = 1
Q_Mu = np.zeros((Num_st,t_T-t_1,L,Q))
Q_Alpha = np.zeros((Num_st,Num_st,L,Q))
#Mu = np.zeros((Num_st,t_T-t_1,L))
Mu = np.transpose(flow_train.min(axis=2),[1,0,2])
Alpha = np.ones((Num_st,Num_st,L)) * 1e-2
for q in range(0,Q):
    for l in range(0,L):
        for i in range(0,Num_st):
            #i
            Oi = np.array([i for i in range(0,Num_st)])
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




endtime = datetime.datetime.now()
Time2 = (endtime - starttime).seconds #5006




np.save('Mu.npy',Mu)
np.save('Alpha.npy',Alpha)



#######################################
#Model testing


#One-step forecast
Num_day_test = flow_test.shape[2]
hat_flow1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    #l = 0
    for i in range(0,Num_st): 
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
    for i in range(0,Num_st): 
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
print('MAE = ', MAE) #MAE =  55.07323279162196
print('RMAE = ', RMAE) #RMAE =  0.06107288218843894

#Two-step forecast
MAE = abs(hat_flow2 - flow_test).mean() 
RMAE = abs(hat_flow2 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 121.93109887339176
print('RMAE = ', RMAE) #RMAE = 0.13521420950132393

#Three-step forecast
MAE = abs(hat_flow3 - flow_test).mean() 
RMAE = abs(hat_flow3 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 167.86825832306397 
print('RMAE = ', RMAE) #RMAE =  0.18615573925964524




#For each layer
l = 0
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 51.93386339919006
print('l = 0, RMAE = ', RMAE) #RMAE = 0.057552818259859695
l = 1
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 58.21260218405389
print('l = 1, RMAE = ', RMAE) #RMAE = 0.06459768244092236


l = 0
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 118.63571811469993
print('l = 0, RMAE = ', RMAE) #RMAE = 0.131471442270743
l = 1
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 125.22647963208354
print('l = 1, RMAE = ', RMAE) #RMAE = 0.13896201270802955


l = 0
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 162.43838830802403
print('l = 0, RMAE = ', RMAE) #RMAE = 0.18001331749299485
l = 1
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 173.29812833810402
print('l = 1, RMAE = ', RMAE) #RMAE = 0.19230642579069573


       

np.save('Proposed_hat_flow1.npy',hat_flow1)
np.save('Proposed_hat_flow2.npy',hat_flow2)
np.save('Proposed_hat_flow3.npy',hat_flow3)




