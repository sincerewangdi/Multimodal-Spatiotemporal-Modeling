#Ablation study: Communities learnt by event frequencies

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




###################################
#Functions



#Parameter initialization is important: least square estimation
def Alphi_i_initialization(i,gamma_i,flow_train,Mu):
    Alpha_i = np.zeros((Num_st,L)) 
    for l in range(0,L):
        Oi = np.argwhere((gamma_i==1))[:,0]
        flow_train_Oi = flow_train[:,Oi,:,:]
        Num_st_Oi = Oi.shape[0]
        Sum_exp = np.zeros((Num_st_Oi,t_T-t_1,t_T-t_1,Num_month_train))
        for t in range(t_1+1, t_T):
            n_jdltj = flow_train_Oi[:t,:,:,l] 
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                        * np.transpose(n_jdltj,(1,0,2))
            Sum_exp[:,t,:t,:] = sum_exp
        Xj = Sum_exp.sum(axis = 2)
        N_idlt = flow_train[:,i,:,l] - np.tile(np.reshape(Mu[i,:,l],(t_T-t_1,1)),(1,Num_month_train))
        Yi = N_idlt
        #    
        #
        Ad = np.zeros((Num_st_Oi,Num_st_Oi,Num_month_train))
        Bd = np.zeros((Num_st_Oi,1,Num_month_train))
        for day in range(0,Num_month_train):
            Ad[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Xj[:,:,day]).T
            Bd[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Yi[:,day]).T
        Alpha_i[Oi,l] = np.array(np.linalg.inv(np.mat(Ad.sum(axis = 2))) * np.mat(Bd.sum(axis = 2)))[:,0]
    return Alpha_i
        



def log_posterior_gamma_i(i,gamma_i,flow_train,Mu):
#Maximize log-posterior to obtain Alpha as a function of Gamma
    #
    Q = 2 #iterative number
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    Q_Alpha_i = np.zeros((Num_st,L,Q))
    Alpha_i = 0.01 * np.ones((Num_st,L)) * np.tile(np.reshape(gamma_i,(Num_st,1)),(1,L))
    #Alpha_i = Alphi_i_initialization(i,gamma_i,flow_train,Mu) #i = 0
    for q in range(0,Q):
        #q
        for l in range(0,L):
            Oi = np.argwhere((gamma_i==1))[:,0]
            flow_train_Oi = flow_train[:,Oi,:,:]
            Num_st_Oi = Oi.shape[0]
            lambda_t = np.zeros((t_T-t_1,Num_month_train))
            Pij = np.zeros((t_T-t_1,Num_st_Oi,t_T-t_1,Num_month_train))
            t = 0
            lambda_t[t,:] = Mu[i,t,l]
            Sum_exp = np.zeros((Num_st_Oi, t_T-t_1,t_T-t_1,Num_month_train))
            for t in range(t_1+1, t_T):
                n_jdltj = flow_train_Oi[:t,:,:,l]
                theta = Alpha_i[Oi,l]
                exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                sum_exp_t = np.tile(np.reshape(theta,(Num_st_Oi,1,1)),(1,t,Num_month_train)) \
                            * np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                            * np.transpose(n_jdltj,(1,0,2))
                lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
                Pij[t,:,:t,:] = sum_exp_t / (0.01+np.tile(np.reshape(lambda_t[t,:],(1,1,Num_month_train)),(Num_st_Oi,t,1)))
                sum_exp = np.tile(np.reshape(exp,(1,t,1)),(Num_st_Oi,1,Num_month_train)) \
                            * np.transpose(n_jdltj,(1,0,2))
                Sum_exp[:,t,:t,:] = sum_exp
            #
            #
            N_idlt = flow_train[:,i,:,l]
            exp = np.array([np.exp( - (t_T - t_j) / delta) for t_j in range(t_1, t_T)])
            #    
            #
            for j in range(0,Num_st_Oi):
                sum1 = (Pij[:,j,:,:] * np.tile(np.reshape(N_idlt,(t_T-t_1,1,Num_month_train)),(1,t_T-t_1,1))).sum()
                sum2 = Sum_exp[j,:,:,:].sum()
                Alpha_i[Oi[j],l] = sum1 / sum2
        Q_Alpha_i[:,:,q] = Alpha_i   
    #
    #
    #log - posterior distribution
    #log P(n_i | alpha_i)
    Pn_id_cir = np.zeros((L,t_T - t_1,Num_month_train))
    for l in range(0,L):
        lambda_t = np.zeros((t_T - t_1,Num_month_train))
        for t in range(t_1+1, t_T):
            n_jdltj = flow_train[:t,:,:,l]
            theta = Alpha_i[:,l]
            exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
            sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_train)) \
                        * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_train)) \
                        * np.transpose(n_jdltj,(1,0,2))
            lambda_t[t,:] = Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
            Lambda_t = lambda_t[t,:]
            n_idlt = flow_train[t,i,:,l]
            Pn_id_cir[l,t,:] = n_idlt * np.log(Lambda_t) - Lambda_t 
            #
        num_00 = np.argwhere(lambda_t <= 0)
        Pn_id_cir[l,num_00[:,0],num_00[:,1]] = 0     
        #
    Pn_id_cir_sum = Pn_id_cir.sum()
    #
    log_Pgamma_i = Pn_id_cir_sum
    return log_Pgamma_i, Alpha_i



#Parameter initialization ########################################
R = 3 #community number

#Mu, Theta
starttime = datetime.datetime.now()
#
Mu = np.zeros((Num_st,t_T-t_1,L))
Theta = np.zeros((Num_st,Num_st,L))
#Inflow only for community 0
for num_c in range(0,R):     
    num_c
    orignal_com = pandas.read_csv('community number_attributes.csv',header=None) #station information
    com = orignal_com.values
    Num_c = np.argwhere(com==num_c)[:,0]
    flow_train_c = flow_train[:,Num_c,:,:] #(234, 30, 15, 2)
    flow_test_c = flow_test[:,Num_c,:,:]
    Inflow_train_c = Inflow_train[:,Num_c,:]
    Inflow_test_c = Inflow_test[:,Num_c,:]
    Num_st_c = flow_train.shape[1]
    #
    num_0 = np.argwhere((flow_train_c==0))
    flow_train_c[num_0[:,0],num_0[:,1],num_0[:,2],num_0[:,3]] = 0.01
    for l in range(0,L):
        Mu_c,Theta_c = initial_eta(flow_train_c,l)
        Mu[Num_c,:,l] = Mu_c[:,:,l]
        for ic1 in range(0,Num_c.shape[0]):
            for ic2 in range(0,Num_c.shape[0]):
                Theta[Num_c[ic1],Num_c[ic2],l] = Theta_c[ic1,ic2,l]



endtime = datetime.datetime.now()
Time1 = (endtime - starttime).seconds #1079

np.save('Initial_Mu.npy',Mu)
np.save('Initial_Alpha.npy',Theta)






#################################################################
#Parameter initialization
#Gamma is given 
Gamma_new = np.zeros((Num_st,Num_st))
for num_c in range(0,R):     
    orignal_com = pandas.read_csv('community number_attributes.csv',header=None) #station information
    com = orignal_com.values
    Num_c = np.argwhere(com==num_c)[:,0]
    for ic1 in range(0,Num_c.shape[0]):
        for ic2 in range(0,Num_c.shape[0]):
            Gamma_new[Num_c[ic1],Num_c[ic2]] = 1



Mu = np.load('Initial_Mu.npy')
Alpha = np.load('Initial_Alpha.npy')




starttime = datetime.datetime.now()
#
#Parameter estimation
All = 80 # total iteration number
All_Mu = 1 * np.zeros((Num_st,t_T-t_1,L,All))
All_Alpha = np.zeros((Num_st,Num_st,L,All))
All_Gamma = np.zeros((Num_st,Num_st,All))
for numa in range(0,All):
    numa
    Gamma = np.ones((Num_st,Num_st)) * Gamma_new
    Q = 10 #iterative number
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    delta = 1
    Q_Mu = np.zeros((Num_st,t_T-t_1,L,Q))
    Q_Alpha = np.zeros((Num_st,Num_st,L,Q))
    Alpha = np.ones((Num_st,Num_st,L)) * 1e-2 * np.tile(np.reshape(Gamma,(Num_st,Num_st,1)),(1,1,L))
    for q in range(0,Q):
        #q
        for l in range(0,L):
            #l
            for i in range(0,Num_st):
                #i
                Oi = np.argwhere((Gamma[i,:]==1))[:,0]
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
    #
    All_Mu[:,:,:,numa] = Mu
    All_Alpha[:,:,:,numa] = Alpha
    #
    #
    #
    # Step 2: Given Tal, Phi, Theta, Mu, Alpha, update Gamma via Bayesian inference (posterior distribution) 
    Gamma_new = np.zeros((Num_st,Num_st))
    for i in range(0,Num_st):
        i
        Log_Pgamma = np.zeros((Num_st,))
        Pgamma_i_cir_sum = np.zeros((Num_st,))
        Log_posterior = np.zeros((Num_st,))
        for j in range(0,Num_st):
            if j == i:
                gamma_i = np.ones((Num_st,)) * Gamma[i,:]
            else:
                gamma_i = np.ones((Num_st,)) * Gamma[i,:]
                gamma_i[j] = 1 - Gamma[i,j]
            #gamma_i
            log_Pgamma_i, Alpha_i = log_posterior_gamma_i(i,gamma_i,flow_train,Mu)
            #Alpha_i
            #log_Pgamma_i
            Log_Pgamma[j] = log_Pgamma_i
            Log_posterior[j] = Log_Pgamma[j]
        #
        #   
        gamma_i_new = np.ones((Num_st,)) * Gamma[i,:]
        num_new = np.argwhere((Log_posterior==Log_posterior.max()))[:,0]
        if abs(Log_posterior[num_new[0]] - Log_posterior[i])<10:
            num_new = i * np.ones((1,))
            #
        num_new
        if len(num_new) == 1:
            if num_new[0] == i:
                gamma_i_new[int(num_new[0])] = Gamma[i,int(num_new[0])]
            else:
                gamma_i_new[int(num_new[0])] = 1 - Gamma[i,int(num_new[0])]
                #
        Gamma_new[i,:] = gamma_i_new
    #
    #
    All_Gamma[:,:,numa] = Gamma_new
    #
    sum(sum(abs(Gamma-Gamma_new)))/ Num_st
    if sum(sum(abs(Gamma-Gamma_new))) / Num_st < 0.1:
        break





endtime = datetime.datetime.now()
Time2 = (endtime - starttime).seconds #65333







np.save('All_Mu.npy',All_Mu)
np.save('All_Alpha.npy',All_Alpha)
np.save('All_Gamma.npy',All_Gamma)





#Model testing
All_Mu = np.load('All_Mu.npy')
All_Alpha = np.load('All_Alpha.npy')
All_Gamma = np.load('All_Gamma.npy')
numa = 35
Mu = All_Mu[:,:,:,numa] 
Alpha = All_Alpha[:,:,:,numa] 
Gamma = All_Gamma[:,:,numa]




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
    for i in range(0,Num_st): #station
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
print('MAE = ', MAE) #MAE =  57.30468002174087
print('RMAE = ', RMAE) #RMAE =  0.06354742212166586

#Two-step forecast
MAE = abs(hat_flow2 - flow_test).mean() 
RMAE = abs(hat_flow2 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 104.99896291322956
print('RMAE = ', RMAE) #RMAE = 0.11643749543759227

#Three-step forecast
MAE = abs(hat_flow3 - flow_test).mean() 
RMAE = abs(hat_flow3 - flow_test).sum() / flow_test.sum() 
print('MAE = ', MAE) #MAE = 138.82068081008117 
print('RMAE = ', RMAE) #RMAE =  0.15394373372835163




#For each layer
l = 0
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 53.7699448834754
print('l = 0, RMAE = ', RMAE) #RMAE = 0.059587553537748104
l = 1
MAE = abs(hat_flow1[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow1[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 60.83941516000636
print('l = 1, RMAE = ', RMAE) #RMAE = 0.0675126187963831


l = 0
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 103.78584828072069
print('l = 0, RMAE = ', RMAE) #RMAE = 0.11501489920233518
l = 1
MAE = abs(hat_flow2[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow2[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 106.21207754573844
print('l = 1, RMAE = ', RMAE) #RMAE = 0.11786200580756159


l = 0
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 0, MAE = ', MAE) #MAE = 138.1409735528603
print('l = 0, RMAE = ', RMAE) #RMAE = 0.15308705774529077
l = 1
MAE = abs(hat_flow3[:,:,:,l] - flow_test[:,:,:,l] ).mean() 
RMAE = abs(hat_flow3[:,:,:,l]  - flow_test[:,:,:,l] ).sum() / flow_test[:,:,:,l].sum() 
print('l = 1, MAE = ', MAE) #MAE = 139.50038806730194
print('l = 1, RMAE = ', RMAE) #RMAE = 0.15480156238790316


       

np.save('Proposed_hat_flow1.npy',hat_flow1)
np.save('Proposed_hat_flow2.npy',hat_flow2)
np.save('Proposed_hat_flow3.npy',hat_flow3)




