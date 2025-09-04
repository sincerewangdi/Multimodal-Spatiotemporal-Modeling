#Model comparison

import seaborn as sns
import matplotlib
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



###################
#Data
Demand0 = np.load('Demand.npy')
st_info0 = np.load('Attr.npy')
num0 = [0,3,4,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21]
Demand = Demand0[:,num0,:,:]
st_info = st_info0[num0,:]


L = 2
Num_month_train = 21
flow_train = Demand[:,:,:Num_month_train,:]
flow_test = Demand[:,:,Num_month_train:,:]


Num_month_train = flow_train.shape[2]
Num_month_test = flow_test.shape[2]
Num_time_pre = flow_train.shape[0]
Num_st = flow_train.shape[1]
t_T = Num_time_pre
t_1 = 0
K = st_info.shape[1] #number of attributes
L = 2 #layers
delta = 1


#Functions
###################################
#Benchmark: S-Hawkes

def ST_Hawkes(flow_train,st_info):
    starttime = datetime.datetime.now()
    #############################################
    #Parameter initialization
    #Mu, Alpha
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3] #layers
    t_T = Num_time_pre
    t_1 = 0
    delta = 10
    #
    #
    #Parameter estimation
    All = 1 # total iteration number
    All_Mu = 1 * np.zeros((Num_st,t_T-t_1,L,All))
    All_Alpha = np.zeros((Num_st,Num_st,L,All))
    #
    numa = 0
    #Step 1: parameter estimation
    #Mu, Alpha
    flow_train_min = flow_train.min(axis = 0)
    Mu = np.zeros((Num_st,t_T-t_1,L))
    for l in range(0,L):       
        mu_l = flow_train_min[:,:,l].mean() * np.ones((Num_st,t_T-t_1))
        Mu[:,:,l] = mu_l
    #
    #
    #least square estimation
    Alpha = np.ones((Num_st,Num_st,L)) 
    for l in range(0,L):
        for i in range(0,Num_st): 
            Oi = np.array([i for i in range(0,Num_st)])
            flow_train_Oi = flow_train[:,Oi,:,:]
            Num_st_Oi = Oi.shape[0]
            Sum_exp = np.zeros((Num_st_Oi,t_T-t_1,t_T-t_1,Num_month_train))
            for t in range(t_1+1, t_T):
                n_jdltj = np.ones((t, Num_st_Oi, Num_month_train)) 
                exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                            * np.transpose(n_jdltj,(1,0,2))
                Sum_exp[:,t,:t,:] = sum_exp
            Xj = Sum_exp.sum(axis = 2)
            N_idlt = flow_train[:,i,:,l] - flow_train[:,:,:,l].min()
            Yi = N_idlt
            Alpha[i,Oi,l] = Yi.sum() / Xj.sum()
            #
            #
    All_Mu[:,:,:,0] = Mu
    All_Alpha[:,:,:,0] = Alpha
    #
    #
    Q = 500 #iterative number
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    Q_Mu = np.zeros((Num_st,t_T-t_1,L,Q))
    Q_Alpha = np.zeros((Num_st,Num_st,L,Q))
    for q in range(0,Q):
        q
        for l in range(0,L):
            for i in range(0,Num_st):
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
                    n_jdltj = np.ones((t, Num_st_Oi, Num_month_train)) #flow_train_Oi[:t,:,:,l] 
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
                #
                #
                N_idlt = flow_train[:,i,:,l] 
                Mu[i,:,l] = (Pii * N_idlt).sum() / (Num_month_train * (t_T-t_1))
                #
                exp = np.array([np.exp( - (t_T - t_j) / delta) for t_j in range(t_1, t_T)])
                #    
                #
                for j in range(0,Num_st_Oi):
                    sum1 = (Pij[:,j,:,:] * np.tile(np.reshape(N_idlt,(t_T-t_1,1,Num_month_train)),(1,t_T-t_1,1))).sum()
                    sum2 = Sum_exp[j,:,:,:].sum()
                    Alpha[i,Oi[j],l] = sum1 / sum2
        Q_Mu[:,:,:,q] = Mu
        Q_Alpha[:,:,:,q] = Alpha
    #
    #    
    #
    #
    All_Mu[:,:,:,numa] = Mu
    All_Alpha[:,:,:,numa] = Alpha
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    #
    return All_Mu,All_Alpha,Time
    
    


All_Mu,All_Alpha,Time = ST_Hawkes(flow_train,st_info)
np.save('All_Mu.npy',All_Mu)
np.save('All_Alpha.npy',All_Alpha)




#Model testing
All_Mu = np.load('All_Mu.npy')
All_Alpha = np.load('All_Alpha.npy')
numa = 0
Mu = All_Mu[:,:,:,numa] 
Alpha = All_Alpha[:,:,:,numa]


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
            n_jdltj = np.ones((t, Num_st, Num_month_test))  
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
            n_jdltj = np.ones((t, Num_st, Num_month_test)) 
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
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
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
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
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
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
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
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda2_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow3[:,i,:,l] = lambda2_t





# Metrics: MAE, RMAE, Time; MAE and RMAE at each layer
#One-step forecast
MAE = abs(hat_flow1[1:,:,:,:] - flow_test[1:,:,:,:]).mean() 
RMAE = abs(hat_flow1[1:,:,:,:] - flow_test[1:,:,:,:]).sum() / flow_test[1:,:,:,:].sum() 
print('MAE = ', MAE) #MAE =  24.882423942341493
print('RMAE = ', RMAE) #RMAE =  0.8717955906219135

#Two-step forecast
MAE = abs(hat_flow2[2:,:,:,:] - flow_test[2:,:,:,:]).mean() 
RMAE = abs(hat_flow2[2:,:,:,:] - flow_test[2:,:,:,:]).sum() / flow_test[2:,:,:,:].sum() 
print('MAE = ', MAE) #MAE = 25.85606246739183
print('RMAE = ', RMAE) #RMAE = 0.8732197658306041

#Three-step forecast
MAE = abs(hat_flow3[3:,:,:,:] - flow_test[3:,:,:,:]).mean() 
RMAE = abs(hat_flow3[3:,:,:,:] - flow_test[3:,:,:,:]).sum() / flow_test[3:,:,:,:].sum() 
print('MAE = ', MAE) #MAE = 26.929376110270294
print('RMAE = ', RMAE) #RMAE =  0.8718623374968669




#For each layer
l = 0
MAE = abs(hat_flow1[1:,:,:,l] - flow_test[1:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[1:,:,:,l]  - flow_test[1:,:,:,l] ).sum() / flow_test[1:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 25.116160477024366
print('l = 0, RMAE = ', RMAE) #RMAE = 0.869571741642175
l = 1
MAE = abs(hat_flow1[1:,:,:,l] - flow_test[1:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[1:,:,:,l]  - flow_test[1:,:,:,l] ).sum() / flow_test[1:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 24.648687407658617
print('l = 1, RMAE = ', RMAE) #RMAE = 0.8740733466391918


l = 0
MAE = abs(hat_flow2[2:,:,:,l] - flow_test[2:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[2:,:,:,l]  - flow_test[2:,:,:,l] ).sum() / flow_test[2:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 26.080648389716043
print('l = 0, RMAE = ', RMAE) #RMAE = 0.8705273737632799
l = 1
MAE = abs(hat_flow2[2:,:,:,l] - flow_test[2:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[2:,:,:,l]  - flow_test[2:,:,:,l] ).sum() / flow_test[2:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 25.63147654506763
print('l = 1, RMAE = ', RMAE) #RMAE = 0.8759764885085374


l = 0
MAE = abs(hat_flow3[3:,:,:,l] - flow_test[3:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[3:,:,:,l]  - flow_test[3:,:,:,l] ).sum() / flow_test[3:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 27.152715311355866
print('l = 0, RMAE = ', RMAE) #RMAE = 0.8688816359550071
l = 1
MAE = abs(hat_flow3[3:,:,:,l] - flow_test[3:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[3:,:,:,l]  - flow_test[3:,:,:,l] ).sum() / flow_test[3:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 26.706036909184718
print('l = 1, RMAE = ', RMAE) #RMAE = 0.8749139334344814

   

np.save('ST_Hawkes_hat_flow1.npy',hat_flow1)
np.save('ST_Hawkes_hat_flow2.npy',hat_flow2)
np.save('ST_Hawkes_hat_flow3.npy',hat_flow3)




######################################################################
#Benchmark: Multimodal S-Hawkes

def multimodalST_Hawkes(flow_train,st_info):
    starttime = datetime.datetime.now()
    #
    Num_st = flow_train.shape[1] #nodes
    Num_time = flow_train.shape[0] #time points
    Num_month_train = flow_train.shape[2] #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = st_info.shape[1] #attributes
    L = flow_train.shape[3] #layers
    #############################################
    #Parameter initialization
    #Mu, Alpha
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    Phi = np.ones((K,))
    for k in range(0,K):
        Phi[k] = st_info[:,k].mean()
    #
    #
    #Parameter estimation
    All = 1 # total iteration number
    All_Mu = 1 * np.zeros((Num_st,t_T-t_1,L,All))
    All_Alpha = np.zeros((Num_st,Num_st,L,All))
    #
    #
    numa = 0  
    #    
    #Step 1: parameter estimation
    #Mu, Alpha
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    delta = 10
    flow_train_min = flow_train.min(axis = 0)
    Mu = np.zeros((Num_st,t_T-t_1,L))
    for l in range(0,L):
        for i in range(0,Num_st):
            Px_id_cir = np.zeros((K,))
            for k in range(0,K):
                x_ik = st_info[i,k]
                Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
            Px_id_cir_sum = np.prod(Px_id_cir)
            #
            mu_l = flow_train_min[i,:,l].mean() * np.ones((t_T-t_1,))
            Mu[i,:,l] = mu_l / Px_id_cir_sum
    #
    #
    #least square estimation
    Alpha = np.zeros((Num_st,Num_st,L)) 
    for l in range(0,L):
        for i in range(0,Num_st): #station
            Oi = np.array([i for i in range(0,Num_st)])
            flow_train_Oi = flow_train[:,Oi,:,:]
            Num_st_Oi = Oi.shape[0]
            Sum_exp = np.zeros((Num_st_Oi,t_T-t_1,t_T-t_1,Num_month_train))
            for t in range(t_1+1, t_T):
                n_jdltj = np.ones((t, Num_st_Oi, Num_month_train)) 
                exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                            * np.transpose(n_jdltj,(1,0,2))
                Sum_exp[:,t,:t,:] = sum_exp
            Xj = Sum_exp.sum(axis = 2)
            N_idlt = flow_train[:,i,:,l] - flow_train[:,:,:,l].min()
            Yi = N_idlt
            #
            Px_id_cir = np.zeros((K,))
            for k in range(0,K):
                x_ik = st_info[i,k]
                Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
            Px_id_cir_sum = np.prod(Px_id_cir)
            #
            Alpha[i,Oi,l] = Alpha[i,Oi,l] = Yi.sum() / (Xj.sum() * Px_id_cir_sum)     
    #
    #
    Q = 500 #iterative number
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    Q_Mu = np.zeros((Num_st,t_T-t_1,L,Q))
    Q_Alpha = np.zeros((Num_st,Num_st,L,Q))
    for q in range(0,Q):
        q
        for l in range(0,L):
            for i in range(0,Num_st):
                Oi = np.array([i for i in range(0,Num_st)])
                flow_train_Oi = flow_train[:,Oi,:,:]
                Num_st_Oi = Oi.shape[0]
                #
                Px_id_cir = np.zeros((K,))
                for k in range(0,K):
                    x_ik = st_info[i,k]
                    Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
                Px_id_cir_sum = np.prod(Px_id_cir)
                #
                lambda_t = np.zeros((t_T-t_1,Num_month_train))
                Pii = np.ones((t_T-t_1,Num_month_train))
                Pij = np.zeros((t_T-t_1,Num_st_Oi,t_T-t_1,Num_month_train))
                t = 0
                lambda_t[t,:] = Mu[i,t,l] *Px_id_cir_sum
                Sum_exp = np.zeros((Num_st_Oi, t_T-t_1,t_T-t_1,Num_month_train))
                for t in range(t_1+1, t_T):
                    n_jdltj = np.ones((t, Num_st_Oi, Num_month_train)) 
                    theta = Alpha[i,Oi,l]
                    exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                    sum_exp_t = np.tile(np.reshape(theta,(Num_st_Oi,1,1)),(1,t,Num_month_train)) \
                                * np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                                * np.transpose(n_jdltj,(1,0,2))
                    lambda_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
                    Pii[t,:] = Mu[i,t,l] * Px_id_cir_sum / lambda_t[t,:]
                    Pij[t,:,:t,:] = sum_exp_t * Px_id_cir_sum \
                                    / (np.tile(np.reshape(lambda_t[t,:],(1,1,Num_month_train)),(Num_st_Oi,t,1)))
                    sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                                * np.transpose(n_jdltj,(1,0,2))
                    Sum_exp[:,t,:t,:] = sum_exp
                #
                #
                N_idlt = flow_train[:,i,:,l] 
                Mu[i,:,l] = (Pii * N_idlt).sum() / (Num_month_train * (t_T-t_1)* Px_id_cir_sum)
                #
                exp = np.array([np.exp( - (t_T - t_j) / delta) for t_j in range(t_1, t_T)])
                #    
                #
                for j in range(0,Num_st_Oi):
                    sum1 = (Pij[:,j,:,:] * np.tile(np.reshape(N_idlt,(t_T-t_1,1,Num_month_train)),(1,t_T-t_1,1))).sum()
                    sum2 = Sum_exp[j,:,:,:].sum()
                    Alpha[i,Oi[j],l] = sum1 / (sum2 * Px_id_cir_sum)
        Q_Mu[:,:,:,q] = Mu
        Q_Alpha[:,:,:,q] = Alpha
    #
    #
    All_Mu[:,:,:,numa] = Mu
    All_Alpha[:,:,:,numa] = Alpha
    #
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    #
    return All_Mu,All_Alpha,Phi,Time






All_Mu,All_Alpha,Phi,Time = multimodalST_Hawkes(flow_train,st_info)
np.save('All_Mu.npy',All_Mu)
np.save('All_Alpha.npy',All_Alpha)
np.save('Phi.npy',Phi)




#Model testing
All_Mu = np.load('All_Mu.npy')
All_Alpha = np.load('All_Alpha.npy')
Phi = np.load('Phi.npy')
numa = 0
Mu = All_Mu[:,:,:,numa] 
Alpha = All_Alpha[:,:,:,numa]


#One-step forecast
Num_day_test = flow_test.shape[2]
hat_flow1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    #l = 0
    for i in range(0,Num_st): #station
        #
        Px_id_cir = np.zeros((K,))
        for k in range(0,K):
            x_ik = st_info[i,k]
            Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
        Px_id_cir_sum = np.prod(Px_id_cir)
        # 
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]* Px_id_cir_sum
        for t in range(t_1+1, t_T):
            n_jdltj = np.ones((t, Num_st, Num_month_test))  
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0))* Px_id_cir_sum
        hat_flow1[:,i,:,l] = lambda_t
            



#Two-step forecast
hat_flow2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
hat_flow_c1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    for i in range(0,Num_st): 
        #
        Px_id_cir = np.zeros((K,))
        for k in range(0,K):
            x_ik = st_info[i,k]
            Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
        Px_id_cir_sum = np.prod(Px_id_cir)
        #
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l] * Px_id_cir_sum
        for t in range(t_1+1, t_T-1):
            n_jdltj = np.ones((t, Num_st, Num_month_test)) 
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
        hat_flow_c1[:,i,:,l] = lambda_t
        #
        lambda1_t = np.zeros((t_T-t_1,Num_month_test))
        lambda1_t[0,:] = lambda_t[0,:]
        lambda1_t[1,:] = lambda_t[1,:]
        for t in range(t_1+2, t_T):
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda1_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
        hat_flow2[:,i,:,l] = lambda1_t





#Three-step forecast
hat_flow3 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
hat_flow_c1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
hat_flow_c2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
Num_month_test = flow_test.shape[2]
for l in range(0,L):
    for i in range(0,Num_st): #station
        #
        Px_id_cir = np.zeros((K,))
        for k in range(0,K):
            x_ik = st_info[i,k]
            Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
        Px_id_cir_sum = np.prod(Px_id_cir)
        #
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l] * Px_id_cir_sum
        for t in range(t_1+1, t_T-2):
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
        hat_flow_c1[:,i,:,l] = lambda_t
        #
        lambda1_t = np.zeros((t_T-t_1,Num_month_test))
        lambda1_t[0,:] = lambda_t[0,:]
        lambda1_t[1,:] = lambda_t[1,:]
        for t in range(t_1+2, t_T-1):
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda1_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
        hat_flow_c2[:,i,:,l] = lambda1_t
        #
        lambda2_t = np.zeros((t_T-t_1,Num_month_test))
        lambda2_t[0,:] = lambda_t[0,:]
        lambda2_t[1,:] = lambda_t[1,:]
        lambda2_t[2,:] = lambda1_t[2,:]
        for t in range(t_1+3, t_T):
            n_jdltj = np.ones((flow_test[:t,:,:,l].shape[0],flow_test[:t,:,:,l].shape[1],flow_test[:t,:,:,l].shape[2]))
            theta = Alpha[i,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda2_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
        hat_flow3[:,i,:,l] = lambda2_t






# Metrics: MAE, RMAE, Time; MAE and RMAE at each layer
#One-step forecast
MAE = abs(hat_flow1[1:,:,:,:] - flow_test[1:,:,:,:]).mean() 
RMAE = abs(hat_flow1[1:,:,:,:] - flow_test[1:,:,:,:]).sum() / flow_test[1:,:,:,:].sum() 
print('MAE = ', MAE) #MAE =  24.88242200335325
print('RMAE = ', RMAE) #RMAE =  0.8717955226863537

#Two-step forecast
MAE = abs(hat_flow2[2:,:,:,:] - flow_test[2:,:,:,:]).mean() 
RMAE = abs(hat_flow2[2:,:,:,:] - flow_test[2:,:,:,:]).sum() / flow_test[2:,:,:,:].sum() 
print('MAE = ', MAE) #MAE = 25.85606055836746
print('RMAE = ', RMAE) #RMAE = 0.8732197013583816

#Three-step forecast
MAE = abs(hat_flow3[3:,:,:,:] - flow_test[3:,:,:,:]).mean() 
RMAE = abs(hat_flow3[3:,:,:,:] - flow_test[3:,:,:,:]).sum() / flow_test[3:,:,:,:].sum() 
print('MAE = ', MAE) #MAE = 26.929374198028164
print('RMAE = ', RMAE) #RMAE =  0.871862275586339




#For each layer
l = 0
MAE = abs(hat_flow1[1:,:,:,l] - flow_test[1:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[1:,:,:,l]  - flow_test[1:,:,:,l] ).sum() / flow_test[1:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 25.116156598798828
print('l = 0, RMAE = ', RMAE) #RMAE = 0.8695716073702452
l = 1
MAE = abs(hat_flow1[1:,:,:,l] - flow_test[1:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[1:,:,:,l]  - flow_test[1:,:,:,l] ).sum() / flow_test[1:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 24.648687407907673
print('l = 1, RMAE = ', RMAE) #RMAE = 0.8740733466480236


l = 0
MAE = abs(hat_flow2[2:,:,:,l] - flow_test[2:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[2:,:,:,l]  - flow_test[2:,:,:,l] ).sum() / flow_test[2:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 26.080644571393524
print('l = 0, RMAE = ', RMAE) #RMAE = 0.8705272463142141
l = 1
MAE = abs(hat_flow2[2:,:,:,l] - flow_test[2:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[2:,:,:,l]  - flow_test[2:,:,:,l] ).sum() / flow_test[2:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 25.631476545341386
print('l = 1, RMAE = ', RMAE) #RMAE = 0.8759764885178932


l = 0
MAE = abs(hat_flow3[3:,:,:,l] - flow_test[3:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[3:,:,:,l]  - flow_test[3:,:,:,l] ).sum() / flow_test[3:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 27.15271148656551
print('l = 0, RMAE = ', RMAE) #RMAE = 0.8688815135624558
l = 1
MAE = abs(hat_flow3[3:,:,:,l] - flow_test[3:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[3:,:,:,l]  - flow_test[3:,:,:,l] ).sum() / flow_test[3:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 26.70603690949082
print('l = 1, RMAE = ', RMAE) #RMAE = 0.8749139334445096

  

np.save('multimodalST_Hawkes_hat_flow1.npy',hat_flow1)
np.save('multimodalST_Hawkes_hat_flow2.npy',hat_flow2)
np.save('multimodalST_Hawkes_hat_flow3.npy',hat_flow3)





##############################################################
#Benchmark: T-Hawkes

#model training
starttime = datetime.datetime.now()
Lambda0 = np.zeros((Num_st, L))
Alpha = np.zeros((Num_st, L))
Beta = np.zeros((Num_st, L))
for l in range(0,L):
    for st in range(0,Num_st):
        #EM Algorithm
        Month = Num_month_train
        T = Num_time_pre
        demand = flow_train[:,st,:,l]
        parameter = [0, 10, 1]
        lambda0 = parameter[0]
        alpha = parameter[1]
        beta = parameter[2]
        #
        Nq = 10 #iterations
        Clambda0 = np.zeros((Nq,))
        Calpha = np.zeros((Nq,))
        Cbeta = np.ones((Nq,)) * beta
        for iq in range(0,Nq):
            #E step
            g = np.zeros((T,T,Month))
            for day in range(0, Month):
                demand_day = demand[:,day]
                for it in range(1,T):
                    for jt in range(0,it):
                        g[it,jt,day] = alpha * np.exp(-beta * (it - jt)) * demand_day[jt]
                #
                #
            pii = np.zeros((T,Month))
            pij = np.zeros((T,T,Month))
            for day in range(0, Month):
                demand_day = demand[:,day]
                for it in range(1,T):
                    Ai = lambda0 + sum(g[it,:,day])
                    if Ai * demand_day[it] > 0:              
                        pii[it,day] = lambda0 / Ai
                        for jt in range(0,it):
                            pij[it,jt,day] = g[it,jt,day] / Ai
            #
            #
            #M step
            lambda0 = (pii * demand).sum() / (T * Month)
            nij = np.zeros((T,T,Month)) 
            for jt in range(0,T):
                nij[:,jt,:] = pij[:,jt,:] * demand
            #
            #   
            n_alpha = np.zeros((T,T,Month))
            exp_alpha = np.zeros((T,T,Month))
            for day in range(0, Month):
                demand_day = demand[:,day]
                for it in range(1,T):
                    for jt in range(0,it):
                        n_alpha[it,jt,day] = nij[it,jt,day]
                        exp_alpha[it,jt,day] = np.exp(-beta * (it - jt)) * demand_day[jt]
                        #
                        #
            alpha = n_alpha.sum() / exp_alpha.sum()
            Clambda0[iq] = lambda0
            Calpha[iq] = alpha
        #
        #
        Lambda0[st,l] = Clambda0[-1]
        Alpha[st,l] = Calpha[-1]
        Beta[st,l] = Cbeta[-1]
        print(st)


endtime = datetime.datetime.now()
Time = (endtime - starttime).seconds
print(Time)


parameter = np.zeros((3,Num_st,L))
parameter[0,:,:] = Lambda0
parameter[1,:,:] = Alpha
parameter[2,:,:] = Beta
np.save('T_Hawkes_parameter.npy',parameter)




#model testing
parameter = np.load('T_Hawkes_parameter.npy')
Lambda0 = parameter[0,:,:] 
Alpha = parameter[1,:,:]
Beta = parameter[2,:,:]

Num_day_test = flow_test.shape[2]
hat_flow1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
for l in range(0,L):
    for st in range(0,Num_st):
        print(st)
        demand = flow_test[:,st,:,l]
        lambda0 = Lambda0[st,l]
        beta = Beta[st,l]
        alpha = Alpha[st,l]
        for day in range(0,Num_day_test):           
            demand_day = demand[:,day]
            Lambda = np.zeros((T,))
            Lambda[0] = lambda0
            for it in range(1, T):
                g = np.zeros((T,T))
                for jt in range(0,it):
                    g[it,jt] = np.exp(-beta * (it - jt)) * demand_day[jt]
                #One-step forecast
                Lambda[it] = lambda0 + alpha * g[it,:].sum()
            hat_flow1[:,st,day,l] = Lambda

                

#Two-step forecast
hat_flow2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
for l in range(0,L):
    for st in range(0,Num_st):
        print(st)
        demand = flow_test[:,st,:,l]
        lambda0 = Lambda0[st,l]
        beta = Beta[st,l]
        alpha = Alpha[st,l]
        for day in range(0,Num_day_test):           
            demand_day = demand[:,day]
            Lambda = np.zeros((T,))
            Lambda[0] = lambda0
            Lambda[1] = lambda0
            for it in range(2, T):
                g1 = np.zeros((T,T))
                for jt in range(0,it-1):
                    g1[it,jt] = np.exp(-beta * (it-1 - jt)) * demand_day[jt]
                Lambda[it-1] = lambda0 + alpha * g1[it,:].sum()
                g2 = np.zeros((T,T))
                for jt in range(0,it-1):
                    g2[it,jt] = np.exp(-beta * (it - jt)) * demand_day[jt]
                g2[it,it-1] = np.exp(-beta * 1) * Lambda[it-1]
                
                Lambda[it] = lambda0 + alpha * g2[it,:].sum()
            hat_flow2[:,st,day,l] = Lambda

            

#Three-step forecast
hat_flow3 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
for l in range(0,L):
    for st in range(0,Num_st):
        print(st)
        demand = flow_test[:,st,:,l]
        lambda0 = Lambda0[st,l]
        beta = Beta[st,l]
        alpha = Alpha[st,l]
        for day in range(0,Num_day_test):           
            demand_day = demand[:,day]
            Lambda = np.zeros((T,))
            Lambda[0] = lambda0
            Lambda[1] = lambda0
            for it in range(2, T):
                #t-2
                g1 = np.zeros((T,T))
                for jt in range(0,it-2):
                    g1[it,jt] = np.exp(-beta * (it - 2 - jt)) * demand_day[jt]
                Lambda[it-2] = lambda0 + alpha * g1[it,:].sum()
                #t-1
                g2 = np.zeros((T,T))
                for jt in range(0,it-2):
                    g2[it,jt] = np.exp(-beta * (it - 1 - jt)) * demand_day[jt]
                g2[it,it-2] = np.exp(-beta * 1) * Lambda[it-2]         
                Lambda[it-1] = lambda0 + alpha * g2[it,:].sum()
                #t
                g3 = np.zeros((T,T))
                for jt in range(0,it-2):
                    g3[it,jt] = np.exp(-beta * (it - jt)) * demand_day[jt]
                g3[it,it-2] = np.exp(-beta * 2) * Lambda[it-2]
                g3[it,it-1] = np.exp(-beta * 1) * Lambda[it-1]
                Lambda[it] = lambda0 + alpha * g3[it,:].sum()               
            hat_flow3[:,st,day,l] = Lambda
            


hat_flow = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3],3))
hat_flow[:,:,:,:,0] = hat_flow1
hat_flow[:,:,:,:,1] = hat_flow2
hat_flow[:,:,:,:,2] = hat_flow3



# Metrics: MAE, RMAE, Time; MAE and RMAE at each layer
#One-step forecast
MAE = abs(hat_flow1 - flow_test).mean() 
RMAE = abs(hat_flow1 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE =  195.4339780595707
print('RMAE = ', RMAE) #RMAE =  0.21672445419739095

#Two-step forecast
MAE = abs(hat_flow2 - flow_test).mean() 
RMAE = abs(hat_flow2 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 196.73066244994118 
print('RMAE = ', RMAE) #RMAE = 0.21816239871225754

#Three-step forecast
MAE = abs(hat_flow3 - flow_test).mean() 
RMAE = abs(hat_flow3 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 198.98213453159673 
print('RMAE = ', RMAE) #RMAE = 0.22065914499395442


#For each layer
l = 0
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 191.73946378856223
print('l = 0, RMAE = ', RMAE) #RMAE = 0.2124846061970073

l = 1
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 199.12849233057915
print('l = 1, RMAE = ', RMAE) #RMAE = 0.22097000700707395






######################################################################
#Benchmark: ConvLSTM

starttime = datetime.datetime.now()
#model training
p = 1
hat_flow = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st * 3,L))
test0 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st * 3,L))
for l in range(0,L):
    l
    trainX = np.zeros((Num_month_train,t_T-t_1-p-3,p * Num_st))
    trainY = np.zeros((Num_month_train,t_T-t_1-p-3,3 * Num_st))
    testX = np.zeros((Num_month_test,t_T-t_1-p-3,p * Num_st))
    testY = np.zeros((Num_month_test,t_T-t_1-p-3,3 * Num_st))
    for i in range(0,Num_st):
        traindata = flow_train[:,i,:,l].T
        testdata = flow_test[:,i,:,l].T
        #cif = CIF[:,i,:,l].T
        for ip in range(0,t_T-t_1-p-3):
            trainX[:,ip,i * p : (i+1) * p] = traindata[:,ip:ip+p]
            trainY[:,ip,i * 3 : (i+1) * 3] = traindata[:,ip+p:ip+p+3]
            testX[:,ip,i * p : (i+1) * p] = testdata[:,ip:ip+p]
            testY[:,ip,i * 3 : (i+1) * 3] = testdata[:,ip+p:ip+p+3]
    #
    #
    #lstm
    Nh = 300
    lstm = Sequential()
    lstm.add(LSTM(Nh, input_shape=(trainX.shape[1], trainX.shape[2]), dropout=0.5, return_sequences=True)) 
    lstm.add(Dense(Num_st * 3, activation='linear'))
    lstm.summary()
    lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = lstm.fit(trainX, trainY, epochs=1000, batch_size=10)
    #model testing
    predY = lstm.predict(testX)
    test0[:,:,:,l] = testY
    hat_flow[:,:,:,l] = predY


endtime = datetime.datetime.now()
Time = (endtime - starttime).seconds 


#
#
#One-step forecast
hat_flow1 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st,L))
test1 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st,L))
ip = 1
for i in range(0,Num_st):
    hat_flow1[:,:,i,:] = hat_flow[:,:,i * 3 + ip-1,:]
    test1[:,:,i,:] = test0[:,:,i * 3 + ip-1,:]

    
    
MAE = abs(hat_flow1 - test1).mean()
RMAE = abs(hat_flow1 - test1).sum() / test1.sum()
print('MAE = ', MAE) #MAE = 5.938279424413358 
print('RMAE = ', RMAE) #MAE = 0.20185151814936958


#Two-step forecast
hat_flow2 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st,L))
test2 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st,L))
ip = 2
for i in range(0,Num_st):
    hat_flow2[:,:,i,:] = hat_flow[:,:,i * 3 + ip-1,:]
    test2[:,:,i,:] = test0[:,:,i * 3 + ip-1,:]
    


    
MAE = abs(hat_flow2 - test2).mean()
RMAE = abs(hat_flow2 - test2).sum() / test2.sum()
print('MAE = ', MAE) #MAE =  6.148967983866284
print('RMAE = ', RMAE) #MAE = 0.20087502925664588




#Three-step forecast
hat_flow3 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st,L))
test3 = np.zeros((Num_month_test,t_T-t_1-p-3,Num_st,L))
ip = 3
for i in range(0,Num_st):
    hat_flow3[:,:,i,:] = hat_flow[:,:,i * 3 + ip-1,:]
    test3[:,:,i,:] = test0[:,:,i * 3 + ip-1,:]
    

    
MAE = abs(hat_flow3 - test3).mean()
RMAE = abs(hat_flow3 - test3).sum() / test3.sum()
print('MAE = ', MAE) #MAE =  6.76825090392441 
print('RMAE = ', RMAE) #MAE = 0.21307072934395413 




#For each layer
l = 0
MAE = abs(hat_flow1[:,:,:,l] - test1[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - test1[:,:,:,l] ).sum() / test1[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 6.24918188837076
print('l = 0, RMAE = ', RMAE) #RMAE = 0.2102360072450046
l = 1
MAE = abs(hat_flow1[:,:,:,l] - test1[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - test1[:,:,:,l] ).sum() / test1[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 5.627376960455957
print('l = 0, RMAE = ', RMAE) #RMAE = 0.19329103318088778


l = 0
MAE = abs(hat_flow2[:,:,:,l] - test2[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - test2[:,:,:,l] ).sum() / test2[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 6.5321668085389
print('l = 0, RMAE = ', RMAE) #RMAE = 0.2111366830526657
l = 1
MAE = abs(hat_flow2[:,:,:,l] - test2[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - test2[:,:,:,l] ).sum() / test2[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 5.765769159193668
print('l = 0, RMAE = ', RMAE) #RMAE = 0.19039164359782537


l = 0
MAE = abs(hat_flow3[:,:,:,l] - test3[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - test3[:,:,:,l] ).sum() / test3[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 7.29835197360787
print('l = 0, RMAE = ', RMAE) #RMAE = 0.22712994101255737
l = 1
MAE = abs(hat_flow3[:,:,:,l] - test3[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - test3[:,:,:,l] ).sum() / test3[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 6.238149834240949
print('l = 0, RMAE = ', RMAE) #RMAE = 0.19868225755525887






######################################################################
#Benchmark: STCM

starttime = datetime.datetime.now()
p = 1
###################################  
hat_flow1 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test,L))
hat_flow2 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test,L))
hat_flow3 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test,L))
test0 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test,L))
#
for l in range(0,L):
    l
    Lambda = flow_train[:,:,:,l].mean(axis = 2)
    U = np.log(Lambda + 0.0001) #100, 20
    #p = 3
    U0 = np.zeros((1, p))
    U1 = np.zeros((1, 1))
    for i in range(0,Num_st):
        u0 = np.zeros((t_T-t_1-p, p))
        u1 = np.zeros((t_T-t_1-p, 1))
        for ip in range(0,t_T-t_1-p):
            u0[ip,:] = U[ip:ip+p,i]
            u1[ip,:] = U[ip+p,i]   
        U0 = np.vstack((U0,u0))
        U1 = np.vstack((U1,u1))       
    #
    #    
    U0 = np.mat(U0[1:,:])
    U1 = np.mat(U1[1:,:])
    Beta = (U0.T * U0).I * U0.T * U1 #parameter
    #
    #One-step forecast
    testU = np.log(flow_test[:,:,:,l]+ 0.0001)
    pred_TestU1 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test))
    for day in range(0,Num_month_test):
        testU_day = testU[:,:,day] 
        U0 = np.zeros((1,p))
        for i in range(0,Num_st):
            u0 = np.zeros((t_T-t_1-p, p))
            for ip in range(0,t_T-t_1-p):
                u0[ip,:] = testU_day[ip:ip+p,i] 
            U0 = np.vstack((U0,u0))
            #
            #      
        U0 = np.mat(U0[1:,:])
        pred_testU_day = np.exp(U0 * Beta)
        pred_TestU1[:,day] = np.array(pred_testU_day)[:,0]
    #Two-step forecast
    pred_TestU2 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test))
    for day in range(0,Num_month_test):
        testU_day = testU[:,:,day] 
        U0 = np.zeros((1,p))
        for i in range(0,Num_st):
            u0 = np.zeros((t_T-t_1-p, p))
            for ip in range(0,t_T-t_1-p):
                u0[ip,:] = testU_day[ip:ip+p,i] 
            U0 = np.vstack((U0,u0))
            #
            #
        U0 = np.mat(np.log(pred_TestU1[:,day]+ 0.0001)).T
        pred_testU_day = np.exp(U0 * Beta)
        pred_TestU2[:,day] = np.array(pred_testU_day)[:,0]
    #Three-step forecast
    pred_TestU3 = np.zeros((Num_st * (t_T-t_1-p), Num_month_test))
    for day in range(0,Num_month_test):
        testU_day = testU[:,:,day] 
        U0 = np.zeros((1,p))
        for i in range(0,Num_st):
            u0 = np.zeros((t_T-t_1-p, p))
            for ip in range(0,t_T-t_1-p):
                u0[ip,:] = testU_day[ip:ip+p,i] 
            U0 = np.vstack((U0,u0))
            #
            #
        U0 = np.mat(np.log(pred_TestU2[:,day]+ 0.0001)).T
        pred_testU_day = np.exp(U0 * Beta)
        pred_TestU3[:,day] = np.array(pred_testU_day)[:,0]      
    #
    #
    #
    testU = flow_test
    TestU = np.zeros((Num_st * (t_T-t_1-p), Num_month_test))
    for day in range(0,Num_month_test):
        testCIF_day = testU[:,:,day,l] 
        U1 = np.zeros((1, 1))
        for i in range(0,Num_st):
            u1 = np.zeros((t_T-t_1-p, 1))
            for ip in range(0,t_T-t_1-p):
                u1[ip,:] = testCIF_day[ip+p,i]   
            U1 = np.vstack((U1,u1))
            #
            #      
        U1 = np.mat(U1[1:,:])
        TestU[:,day] = np.array(U1)[:,0]
    #
    #
    hat_flow1[:,:,l] = pred_TestU1
    hat_flow2[:,:,l] = pred_TestU2
    hat_flow3[:,:,l] = pred_TestU3
    test0[:,:,l] = TestU
    #


endtime = datetime.datetime.now()
Time = (endtime - starttime).seconds 


#demand error for each OD at each time and day
#One-step forecast
MAE = abs(hat_flow1 - test0).mean()
RMAE = abs(hat_flow1 - test0).sum() / test0.sum()
print('MAE = ', MAE) #MAE =  8.593212042512663
print('RMAE = ', RMAE) #MAE =  0.3010769523620896

#Two-step forecast
MAE = abs(hat_flow2 - test0).mean()
RMAE = abs(hat_flow2 - test0).sum() / test0.sum()
print('MAE = ', MAE) #MAE = 10.846794123233462 
print('RMAE = ', RMAE) #MAE = 0.3800348113564361 

#Three-step forecast
MAE = abs(hat_flow3 - test0).mean()
RMAE = abs(hat_flow3 - test0).sum() / test0.sum()
print('MAE = ', MAE) #MAE = 13.112154695031174 
print('RMAE = ', RMAE) #MAE = 0.45940534865772026 





#For each layer
l = 0
MAE = abs(hat_flow1[:,:,l] - test0[:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,l]  - test0[:,:,l] ).sum() / test0[:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 9.24870921421486
print('l = 0, RMAE = ', RMAE) #RMAE = 0.32020882278976714
l = 1
MAE = abs(hat_flow1[:,:,l] - test0[:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,l]  - test0[:,:,l] ).sum() / test0[:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 7.937714870810463
print('l = 0, RMAE = ', RMAE) #RMAE = 0.2814813173232593


l = 0
MAE = abs(hat_flow2[:,:,l] - test0[:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,l]  - test0[:,:,l] ).sum() / test0[:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 12.401141188493863
print('l = 0, RMAE = ', RMAE) #RMAE = 0.42935232682136115
l = 1
MAE = abs(hat_flow2[:,:,l] - test0[:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,l]  - test0[:,:,l] ).sum() / test0[:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 9.292447057973064
print('l = 0, RMAE = ', RMAE) #RMAE = 0.329521818508938


l = 0
MAE = abs(hat_flow3[:,:,l] - test0[:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,l]  - test0[:,:,l] ).sum() / test0[:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 15.27855246695248
print('l = 0, RMAE = ', RMAE) #RMAE = 0.5289740639542708
l = 1
MAE = abs(hat_flow3[:,:,l] - test0[:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,l]  - test0[:,:,l] ).sum() / test0[:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 10.945756923109872
print('l = 0, RMAE = ', RMAE) #RMAE = 0.3881502583504327




##############################################################
#Benchmark: MSHP

###################################
#Parameter initialization
R = 3 #community number
km = KMeans(n_clusters=R)
km.fit(st_info)
y_pred1 = km.labels_
tal0 = 1/int(R/2)
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
    for r in range(0,R): 
        for i in range(0,Num_st): 
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
    for r in range(0,R): 
        for k in range(0,K):
            sum_phi_0 = np.zeros((Num_st,))
            sum_phi_1 = np.zeros((Num_st,))
            for i in range(0,Num_st): 
                sum_phi_0[i] = hat_tal_ird[i,r] * st_info[i,k]
                sum_phi_1[i] = hat_tal_ird[i,r]
            Phi[r,k] = sum_phi_0.sum() / sum_phi_1.sum()
    Q_Phi[:,:,q] = Phi
    Q_Tal[:,:,q] = hat_tal_ird
            
        


#Model training
starttime = datetime.datetime.now()
R = 3 #community number

Q = 800 #iterative number
l = 0 #layer 1
t_T = Num_time_pre
t_1 = 0

Q_Mu = 1 * np.zeros((Num_st,t_T-t_1,L,Q))
Q_Theta = np.zeros((R,Num_st,L,Q))
Q_Tal = np.zeros((Num_st,R,Q))
Q_Phi = np.ones((R,K,Q))

Mu = 0.1 * np.ones((Num_st,t_T-t_1,L))
Theta = np.ones((R,Num_st,L)) * 1e-2
Tal = hat_tal_ird
#Model training
for q in range(0,Q): 
    q
    # E step
    Pidr = np.zeros((Num_st,Num_month_train,R))
    Px_id_cir_sum = np.zeros((R,Num_st,Num_month_train))
    Pn_id_cir_sum = np.zeros((R,Num_st,Num_month_train))
    for r in range(0,R): 
        for i in range(0,Num_st): 
            Pc_ir = Tal[i,r]
            #
            Px_id_cir = np.zeros((K,))
            for k in range(0,K):
                x_ik = st_info[i,k]
                Px_id_cir[k] = poisson.pmf(x_ik, Phi[r,k])   
            Px_id_cir_sum[r,i,:] = np.prod(Px_id_cir)
    #
    #
    #
    for r in range(0,R):         
        for i in range(0,Num_st): 
            Pn_id_cir = np.zeros((L,t_T - t_1,Num_month_train))
            for l in range(0,L):
                lambda_t = np.zeros((t_T - t_1,Num_month_train))
                for t in range(t_1, t_T):
                    n_jdltj = flow_train[:t,:,:,l]
                    theta = Theta[r,:,l]
                    exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                    sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_train)) \
                                * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_train)) \
                                * np.transpose(n_jdltj,(1,0,2))
                    lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
                    Lambda_t = lambda_t[t,:]
                    n_idlt = flow_train[t,i,:,l]
                    sum_xik = np.zeros((Num_month_train,))
                    for d in range(0,Num_month_train):
                        sum_xik0 = 0
                        for ix in range(1,int(n_idlt[d])+1):
                            sum_xik0 = np.log(ix) + sum_xik0
                        sum_xik[d] = sum_xik0
                    Pn_id_cir[l,t,:] = n_idlt * np.log(Lambda_t) - Lambda_t - sum_xik
                    #
                num_00 = np.argwhere(lambda_t == 0)
                Pn_id_cir[l,num_00[:,0],num_00[:,1]] = 0     
            Pn_id_cir_sum[r,i,:] = np.sum(np.sum(Pn_id_cir,axis = 0),axis = 0)
    #
    #
    #
    hat_tal_ird = np.zeros((Num_st,R,Num_month_train))
    for i in range(0,Num_st): 
        for d in range(0,Num_month_train):
            log_0 = sorted(Pn_id_cir_sum[:,i,d],reverse=True)
            exp_value = np.zeros((R,))
            max0 = 700
            if log_0[0] - log_0[2] < max0:
                exp_value[np.argwhere(Pn_id_cir_sum[:,i,d] == log_0[0])] = np.exp(log_0[0] - log_0[2])
                exp_value[np.argwhere(Pn_id_cir_sum[:,i,d] == log_0[1])] = np.exp(log_0[1] - log_0[2])
            else:
                if log_0[0] - log_0[1] < 500:
                    exp_value[np.argwhere(Pn_id_cir_sum[:,i,d] == log_0[0])] = np.exp(log_0[0] - log_0[1])
                else:
                    exp_value[np.argwhere(Pn_id_cir_sum[:,i,d] == log_0[0])] = np.exp(max0)
            #
            #
            sum0 = Px_id_cir_sum[:,i,d] * exp_value * Tal[i,:]
            sum1 = sum(Px_id_cir_sum[:,i,d] * exp_value * Tal[i,:])
            if sum1 == 0:
                hat_tal_ird[i,:,d] = exp_value / exp_value.sum()
            else:
                hat_tal_ird[i,:,d] = sum0/sum1
    #
    #
    #
    # M step
    sum_tal0 = hat_tal_ird.sum(axis = 2)
    sum_tal1 = hat_tal_ird.sum(axis = 2).sum(axis = 1)
    #
    #Tal
    Tal =  sum_tal0 / np.tile(np.reshape(sum_tal1,(Num_st,1)),(1,R))
    Q_Tal[:,:,q] = Tal
    #
    #Phi
    for r in range(0,R): 
        for k in range(0,K):
            sum_phi0 = (hat_tal_ird.sum(axis = 2)[:,r] * st_info[:,k]).sum()
            sum_phi1 = (hat_tal_ird.sum(axis = 2)[:,r]).sum()
            Phi[r,k] = sum_phi0 / sum_phi1
    #
    #
    Q_Phi[:,:,q] = Phi
    #
    #
    #
    for l in range(0,L):
        #Mu
        Pii = np.ones((R,Num_st,t_T-t_1,Num_month_train))
        Pij = np.zeros((R,Num_st,t_T-t_1,Num_st,Num_month_train))
        Sum_exp = np.zeros((R,Num_st,Num_st,t_T-t_1,Num_month_train))
        lambda_t = np.zeros((Num_st,R,t_T-t_1,Num_month_train))
        for t in range(t_1+1, t_T):
            n_jdltj = flow_train[:t,:,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)])
            sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_train)) \
                * np.transpose(n_jdltj,(1,0,2))
            theta = Theta[:,:,l]
            sum_exp_t = np.tile(np.reshape(theta,(R,Num_st,1,1)),(1,1,t,Num_month_train)) \
                        * np.tile(np.reshape(exp,(1,1,t,1)),(R,Num_st,1,Num_month_train)) \
                        * np.tile(np.reshape(np.transpose(n_jdltj,(1,0,2)),(1,Num_st,t,Num_month_train)),(R,1,1,1))
            for i in range(0,Num_st): 
                Sum_exp[:,i,:,t,:] = np.tile(np.reshape(sum_exp.sum(axis = 1),(1,Num_st,Num_month_train)),(R,1,1))
                lambda_t[i,:,0,:] = Mu[i,0,l]
                lambda_t[i,:,t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 1).sum(axis = 1)
                Pii[:,i,t,:] = Mu[i,t,l] / lambda_t[i,:,t,:]
                Pij[:,i,t,:,:] = (sum_exp_t / np.tile(np.reshape(lambda_t[i,:,t,:],(R,1,1,Num_month_train)),(1,Num_st,t,1))).sum(axis = 2)
        #       
        #
        #
        for i in range(0,Num_st): 
            N_idlt = flow_train[:,i,:,l]
            for t in range(t_1, t_T):
                Mu[i,t,l] = ((hat_tal_ird[i,:,:] * Pii[:,i,t,:]).sum(axis = 0) * N_idlt[t,:]).sum() / hat_tal_ird[i,:,:].sum()
        #
        #
        #
        N_idlt = flow_train[:,:,:,l]
        for r in range(0,R):
            for j in range(0,Num_st):  
                sum1 = ((Pij[r,:,:,j,:] * N_idlt.transpose(1,0,2)).sum(axis = 1) * hat_tal_ird[:,r,:]).sum()
                sum2 = (Sum_exp[r,:,j,:,:].sum(axis = 1) * hat_tal_ird[:,r,:]).sum()
                Theta[r,j,l] = sum1 / sum2
    #
    #
    #       
    Q_Mu[:,:,:,q] = Mu
    Q_Theta[:,:,:,q] = Theta



endtime = datetime.datetime.now()
Time1 = (endtime - starttime).seconds 


np.save('Mu.npy',Mu)
np.save('Theta.npy',Theta)
np.save('Tal.npy',Tal)




#model testing
Mu = np.load('Mu.npy')
Theta = np.load('Theta.npy')
Tal = np.load('Tal.npy')

#One-step forecast
Num_day_test = flow_test.shape[2]
hat_flow1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
for i in range(0,Num_st): 
    r = np.argwhere(Tal[i,:] ==Tal[i,:].max())[0,0]
    for l in range(0,L):
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]
        for t in range(t_1+1, t_T):
            n_jdltj = flow_test[:t,:,:,l]
            theta = Theta[r,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow1[:,i,:,l] = lambda_t

                

#Two-step forecast
flow_test = np.load('flow_test.npy')
hat_flow2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
hat_flow_c1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
for i in range(0,Num_st): 
    r = np.argwhere(Tal[i,:] ==Tal[i,:].max())[0,0]
    for l in range(0,L):
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]
        for t in range(t_1+1, t_T-1):
            n_jdltj = flow_test[:t,:,:,l]
            theta = Theta[r,:,l]
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
            theta = Theta[r,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda1_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow2[:,i,:,l] = lambda1_t

            


#Three-step forecast
flow_test = np.load('flow_test.npy')
hat_flow3 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
hat_flow_c1 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
hat_flow_c2 = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
for i in range(0,Num_st): 
    r = np.argwhere(Tal[i,:] ==Tal[i,:].max())[0,0]
    for l in range(0,L):
        lambda_t = np.zeros((t_T-t_1,Num_month_test))
        t = 0
        lambda_t[t,:] = Mu[i,t,l]
        for t in range(t_1+1, t_T-1):
            n_jdltj = flow_test[:t,:,:,l]
            theta = Theta[r,:,l]
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
            theta = Theta[r,:,l]
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
            theta = Theta[r,:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
            lambda2_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
        hat_flow3[:,i,:,l] = lambda2_t

       


            
            

# Metrics: MAE, RMAE, Time; MAE and RMAE at each layer
#One-step forecast
MAE = abs(hat_flow1 - flow_test).mean() 
RMAE = abs(hat_flow1 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 6.284114729592401
print('RMAE = ', RMAE) #RMAE =  0.22616082049732128

#Two-step forecast
MAE = abs(hat_flow2 - flow_test).mean() 
RMAE = abs(hat_flow2 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 8.140512911013488 
print('RMAE = ', RMAE) #RMAE = 0.29297127096584097

#Three-step forecast
MAE = abs(hat_flow3 - flow_test).mean() 
RMAE = abs(hat_flow3 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 9.390713808916436 
print('RMAE = ', RMAE) #RMAE = 0.33796511226615056




#For each layer
l = 0
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 6.472579971735169
print('l = 0, RMAE = ', RMAE) #RMAE = 0.23006854933326065
l = 1
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 6.0956494874496325
print('l = 1, RMAE = ', RMAE) #RMAE = 0.22215419162455785


l = 0
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 9.325466174595903
print('l = 0, RMAE = ', RMAE) #RMAE = 0.3314746954096739
l = 1
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 6.955559647431072
print('l = 1, RMAE = ', RMAE) #RMAE = 0.2534933699768793


l = 0
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 10.986888204847014
print('l = 0, RMAE = ', RMAE) #RMAE = 0.39053011967625473
l = 1
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 7.794539412985855
print('l = 1, RMAE = ', RMAE) #RMAE = 0.2840697461267758



