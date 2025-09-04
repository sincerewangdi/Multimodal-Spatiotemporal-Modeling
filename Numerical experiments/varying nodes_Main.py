# Table B2, Varying nodes:
import pandas
import math
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from scipy.stats import poisson
from scipy.stats import norm
import datetime
import scipy
from sklearn.cluster import KMeans
import csv
######################################################################
# functions

#data generation
def data_generating(Num_st,R):
    import pandas
    import math
    import numpy as np
    import seaborn as sns
    import matplotlib
    from matplotlib import pyplot
    from scipy.stats import poisson
    from scipy.stats import norm
    import datetime
    import scipy
    #Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    #
    ##True parameter setting
    rTal = np.load('rTal.npy')
    rTheta = np.load('rTheta.npy')
    rGamma = np.load('rGamma.npy')
    rPhi = np.load('rPhi.npy')
    rMu = np.load('rMu.npy')
    rAlpha = np.load('rAlpha.npy')
    #
    ##Data
    #Attributes data
    st_info = np.zeros((Num_st,K))
    for i in range(0,Num_st):
        numi = np.argwhere(rTal[i,:] == 1)[0,0]
        phii = rPhi[numi,:]
        for k in range(0,K):
            st_info[i,k] = np.random.poisson(phii[k])
    #
    #    
    #training data
    delta = 0.2
    flow_train = np.zeros((Num_time,Num_st,Num_month_train,L))
    flow_train_CIF = np.zeros((Num_time,Num_st,Num_month_train,L))
    for day in range(0,Num_month_train):
        for l in range(0,L):
            lambda_t = np.zeros((t_T-t_1,Num_st))
            for i in range(0,Num_st):
                lambda_t[0,i] = np.random.normal()*0.1 +rMu[i,0,l]
            lam0_t = np.zeros((t_T-t_1,Num_st))
            lam0_t[0,:] = rMu[:,0,l]
            flow_train_CIF[0,:,day,l] = lambda_t[0,:]
            flow_train[0,:,day,l] = lambda_t[0,:]
            for t in range(t_1+1, t_T):
                for i in range(0,Num_st):
                    n_jdltj = flow_train[:t,:,day,l]
                    #n_jdltj
                    alpha = rAlpha[i,:,l]
                    exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                    sum_exp_t = np.tile(np.reshape(alpha,(Num_st,1)),(1,t)) \
                                * np.tile(np.reshape(exp,(1,t)),(Num_st,1)) \
                                * np.transpose(n_jdltj,(1,0))
                    lam0_t[t,i] = rMu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
                    lambda_t[t,i] = round(np.random.poisson(lam0_t[t,i],10).mean())
                flow_train_CIF[t,:,day,l] = lam0_t[t,:]
                flow_train[t,:,day,l] = lambda_t[t,:]
    #
    #
    #test data
    flow_test = np.zeros((Num_time,Num_st,Num_month_train,L))
    flow_test_CIF = np.zeros((Num_time,Num_st,Num_month_train,L))
    for day in range(0,Num_month_train):
        for l in range(0,L):
            lambda_t = np.zeros((t_T-t_1,Num_st))
            for i in range(0,Num_st):
                lambda_t[0,i] = np.random.normal()*0.1 +rMu[i,0,l]
            lam0_t = np.zeros((t_T-t_1,Num_st))
            lam0_t[0,:] = rMu[:,0,l]
            flow_test_CIF[0,:,day,l] = lambda_t[0,:]
            flow_test[0,:,day,l] = lambda_t[0,:]
            for t in range(t_1+1, t_T):
                for i in range(0,Num_st):
                    n_jdltj = flow_test[:t,:,day,l]
                    #n_jdltj
                    alpha = rAlpha[i,:,l]
                    exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                    sum_exp_t = np.tile(np.reshape(alpha,(Num_st,1)),(1,t)) \
                                * np.tile(np.reshape(exp,(1,t)),(Num_st,1)) \
                                * np.transpose(n_jdltj,(1,0))
                    lam0_t[t,i] = rMu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)
                    lambda_t[t,i] = round(np.random.poisson(lam0_t[t,i],10).mean())
                flow_test_CIF[t,:,day,l] = lam0_t[t,:]
                flow_test[t,:,day,l] = lambda_t[t,:]
    return st_info, flow_train, flow_train_CIF, flow_test, flow_test_CIF





###################################
#Proposed method
def model_proposed(flow_train,st_info,R):
    starttime = datetime.datetime.now()
    #
    Num_st = flow_train.shape[1] #nodes
    Num_time = flow_train.shape[0] #time points
    Num_month_train = flow_train.shape[2] #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = st_info.shape[1] #attributes
    L = flow_train.shape[3] #layers
    #R = 3 #communities
    #############################################
    #Parameter initialization
    #Gamma is given 
    Gamma = np.zeros((Num_st,Num_st))
    for num_c in range(0,R):     
        orignal_com = pandas.read_csv('community number_attributes.csv',header=None) #station information
        com = orignal_com.values
        Num_c = np.argwhere(com==num_c)[:,0]
        for ic1 in range(0,Num_c.shape[0]):
            for ic2 in range(0,Num_c.shape[0]):
                Gamma[Num_c[ic1],Num_c[ic2]] = 1
    #
    #
    #Mu, Alpha
    delta = 0.2
    flow_train_min = flow_train.min(axis = 0)
    Mu = np.zeros((Num_st,t_T-t_1,L))
    for l in range(0,L):       
        mu_l = flow_train_min[:,:,l].mean() * np.ones((Num_st,t_T-t_1))
        Mu[:,:,l] = mu_l
    #
    #
    #least square estimation
    Alpha = np.zeros((Num_st,Num_st,L)) 
    for l in range(0,L):
        for i in range(0,Num_st): #station
            Oi = np.argwhere((Gamma[i,:]==1))[:,0]
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
            N_idlt = flow_train[:,i,:,l] - flow_train[:,:,:,l].min()
            Yi = N_idlt
            #    
            #
            Ad = np.zeros((Num_st_Oi,Num_st_Oi,Num_month_train))
            Bd = np.zeros((Num_st_Oi,1,Num_month_train))
            for day in range(0,Num_month_train):
                Ad[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Xj[:,:,day]).T
                Bd[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Yi[:,day]).T
            Alpha[i,Oi,l] = np.array(np.linalg.inv(np.mat(Ad.sum(axis = 2))) * np.mat(Bd.sum(axis = 2)))[:,0]
    #
    #
    #Parameter estimation
    All = 11 # total iteration number
    All_Theta = np.ones((R,R,All))
    All_Phi = np.ones((R,K,All))
    All_Tal = np.zeros((Num_st,R,All))
    All_Z = np.zeros((Num_st,R,All))
    All_Mu = 1 * np.zeros((Num_st,t_T-t_1,L,All))
    All_Alpha = np.zeros((Num_st,Num_st,L,All))
    All_Gamma = np.zeros((Num_st,Num_st,All))
    #
    All_Gamma[:,:,0] = Gamma
    All_Mu[:,:,:,0] = Mu
    All_Alpha[:,:,:,0] = Alpha
    #
    #
    #
    for numa in range(0,All-1):
        #numa
        Gamma = np.ones((Num_st,Num_st)) * All_Gamma[:,:,numa]
        #
        # Step 1: Given Gamma, estimate Tal, Phi, Theta, Mu, Alpha via maximum likelihood estimation (MLE)
        tal0 = 1/int(R/2+1)
        Tal = (1-tal0)/(R-1) * np.ones((Num_st,R))
        for i in range(0,Num_st):
            Tal[i,int(com[i])] = tal0
        #
        #    
        Z = np.zeros((Num_st,R))
        for i in range(0,Num_st):
            Z[i,np.argwhere((Tal[i,:] == Tal[i,:].max()))] = 1
        #
        #
        #(1) Given Gamma, estimate Tal, Phi, Theta
        K = st_info.shape[1]
        Q = 20 #iterative number
        Phi = np.ones((R,K))
        Theta = 0.5 * np.ones((R,R))
        Pidr = np.zeros((Num_st,R))
        Q_Tal = np.zeros((Num_st,R,Q))
        Q_Phi = np.ones((R,K,Q))
        Q_Theta = np.ones((R,R,Q))
        Q_Tal[:,:,0] = Tal
        Q_Theta[:,:,0] = np.eye((R)) * 0.6 + np.ones((R,R)) * 0.4
        for q in range(1,Q):  
            Tal = np.ones((Num_st,R)) * Q_Tal[:,:,q-1]
            Theta = np.ones((R,R)) * Q_Theta[:,:,q-1]
            Phi = np.ones((R,K)) * Q_Phi[:,:,q-1]
            for i in range(0,Num_st): 
                for r in range(0,R): #community
                    Pc_ir = Tal[i,r]
                    Pgamma_i_cir = np.zeros((Num_st,R))
                    for j in range(0,Num_st):
                        for r1 in range(0,R):
                            gamma_ij = Gamma[i,j]
                            theta_rr1 = Theta[r,r1]
                            Bernoulli = theta_rr1 ** Gamma[i,j] * (1-theta_rr1) ** (1-Gamma[i,j])
                            Pgamma_i_cir[j,r1] = Bernoulli**Z[j,r1]     
                    Pgamma_i_cir_sum = np.prod(Pgamma_i_cir)
                    #
                    Px_id_cir = np.zeros((K,))
                    for k in range(0,K):
                        x_ik = st_info[i,k]
                        Px_id_cir[k] = poisson.pmf(x_ik, Phi[r,k])   
                    Px_id_cir_sum = np.prod(Px_id_cir)    
                    Pidr[i,r] = Pc_ir * Pgamma_i_cir_sum * Px_id_cir_sum
            hat_tal_ird = Pidr / np.tile(np.reshape(Pidr.sum(axis = 1),(Num_st,1)),(1,R))
            if len(np.argwhere((Pidr.sum(axis = 1)==0))[:,0])>0:
                hat_tal_ird[np.argwhere((Pidr.sum(axis = 1)==0))[:,0],:] = 1/R  
            for r in range(0,R):
                for k in range(0,K):
                    sum_phi_0 = np.zeros((Num_st,))
                    sum_phi_1 = np.zeros((Num_st,))
                    for i in range(0,Num_st): 
                        sum_phi_0[i] = hat_tal_ird[i,r] * st_info[i,k]
                        sum_phi_1[i] = hat_tal_ird[i,r]
                    Phi[r,k] = sum_phi_0.sum() / sum_phi_1.sum()
            for r in range(0,R): 
                for r1 in range(0,R):
                    sum_theta_0 = np.zeros((Num_st,Num_st))
                    sum_theta_1 = np.zeros((Num_st,Num_st))
                    for i in range(0,Num_st):
                        for j in range(0,Num_st): 
                            sum_theta_0[i,j] = hat_tal_ird[i,r] * hat_tal_ird[j,r1] * Gamma[i,j]
                            sum_theta_1[i,j] = hat_tal_ird[i,r] * hat_tal_ird[j,r1]
                    Theta[r,r1] = sum_theta_0.sum() / sum_theta_1.sum()     
            Q_Phi[:,:,q] = Phi
            Q_Tal[:,:,q] = hat_tal_ird
            Q_Theta[:,:,q] = Theta
            Z = np.zeros((Num_st,R))
            for i in range(0,Num_st):
                Z[i,np.argwhere((hat_tal_ird[i,:] == hat_tal_ird[i,:].max()))] = 1
        #        
        #            
        #
        #Step 1: parameter estimation
        #Mu, Alpha
        Num_time_pre = flow_train.shape[0]
        Num_st = flow_train.shape[1]
        Num_month_train = flow_train.shape[2]
        L = flow_train.shape[3]
        t_T = Num_time_pre
        t_1 = 0
        delta = 0.2
        flow_train_min = flow_train.min(axis = 0)
        Mu = np.zeros((Num_st,t_T-t_1,L))
        for l in range(0,L):       
            mu_l = flow_train_min[:,:,l].mean() * np.ones((Num_st,t_T-t_1))
            Mu[:,:,l] = mu_l
        #
        #
        #least square estimation
        Alpha = np.zeros((Num_st,Num_st,L)) 
        for l in range(0,L):
            for i in range(0,Num_st): #station
                Oi = np.argwhere((Gamma[i,:]==1))[:,0]
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
                N_idlt = flow_train[:,i,:,l] - flow_train[:,:,:,l].min()
                Yi = N_idlt
                #    
                #
                Ad = np.zeros((Num_st_Oi,Num_st_Oi,Num_month_train))
                Bd = np.zeros((Num_st_Oi,1,Num_month_train))
                for day in range(0,Num_month_train):
                    Ad[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Xj[:,:,day]).T
                    Bd[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Yi[:,day]).T
                Alpha[i,Oi,l] = np.array(np.linalg.inv(np.mat(Ad.sum(axis = 2))) * np.mat(Bd.sum(axis = 2)))[:,0]
        #
        #
        Q = 20 #iterative number
        Num_time_pre = flow_train.shape[0]
        Num_st = flow_train.shape[1]
        Num_month_train = flow_train.shape[2]
        L = flow_train.shape[3]
        t_T = Num_time_pre
        t_1 = 0
        delta = 0.2
        Q_Mu = np.zeros((Num_st,t_T-t_1,L,Q))
        Q_Alpha = np.zeros((Num_st,Num_st,L,Q))
        for q in range(0,Q):
            #q
            for l in range(0,L):
                for i in range(0,Num_st):
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
        #
        All_Theta[:,:,numa] = Theta
        All_Phi[:,:,numa] = Phi
        All_Tal[:,:,numa] = hat_tal_ird
        All_Z[:,:,numa] = Z
        All_Mu[:,:,:,numa] = Mu
        All_Alpha[:,:,:,numa] = Alpha
        #
        #    
        #
        #
        # Step 2: Given Tal, Phi, Theta, Mu, Alpha, update Gamma via Bayesian inference (posterior distribution) 
        Gamma_new = np.zeros((Num_st,Num_st))
        for i in range(0,Num_st):
            #i
            Log_Pgamma = np.zeros((Num_st,))
            Pgamma_i_cir_sum = np.zeros((Num_st,))
            Log_posterior = np.zeros((Num_st,))
            for j in range(0,Num_st):
                if j == i:
                    gamma_i = np.ones((Num_st,)) * Gamma[i,:]
                else:
                    gamma_i = np.ones((Num_st,)) * Gamma[i,:]
                    gamma_i[j] = 1 - Gamma[i,j]
                log_Pgamma_i, Alpha_i = log_posterior_gamma_i(i,gamma_i,flow_train,Mu)
                Log_Pgamma[j] = log_Pgamma_i
                Pgamma_i_cir_sum[j] = log_gamma_i_z(gamma_i,Theta,Z,i)
                Log_posterior[j] = Log_Pgamma[j] + Pgamma_i_cir_sum[j] 
            #
            #   
            gamma_i_new = np.ones((Num_st,)) * Gamma[i,:]
            num_new = np.argwhere((Log_posterior==Log_posterior.max()))[:,0]
            if len(num_new) == 1:
                if num_new[0] == i:
                    gamma_i_new[int(num_new[0])] = Gamma[i,int(num_new[0])]
                else:
                    gamma_i_new[int(num_new[0])] = 1 - Gamma[i,int(num_new[0])]
                    #
            Gamma_new[i,:] = gamma_i_new
        #
        #
        if (Gamma_new == Gamma).mean() == 1:
            break
        else:
            All_Gamma[:,:,numa+1] = Gamma_new
    #
    #
    #
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    #
    return All_Theta, All_Phi,All_Tal,All_Z,All_Mu,All_Alpha,All_Gamma,Time






def Alphi_i_initialization(i,gamma_i,flow_train,Mu):
    import numpy as np
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
        N_idlt = flow_train[:,i,:,l] - flow_train[:,:,:,l].min()
        Yi = N_idlt
        #
        Ad = np.zeros((Num_st_Oi,Num_st_Oi,Num_month_train))
        Bd = np.zeros((Num_st_Oi,1,Num_month_train))
        for day in range(0,Num_month_train):
            Ad[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Xj[:,:,day]).T
            Bd[:,:,day] = np.mat(Xj[:,:,day]) * np.mat(Yi[:,day]).T
        Alpha_i[Oi,l] = np.array(np.linalg.inv(np.mat(Ad.sum(axis = 2))) * np.mat(Bd.sum(axis = 2)))[:,0]
    return Alpha_i
        



def log_posterior_gamma_i(i,gamma_i,flow_train,Mu):
    import numpy as np
    #Maximize log-posterior to obtain Alpha as a function of Gamma
    #
    Q = 1 #iterative number
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    Q_Alpha_i = np.zeros((Num_st,L,Q))
    Alpha_i = Alphi_i_initialization(i,gamma_i,flow_train,Mu)
    for q in range(0,Q):
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
                Pij[t,:,:t,:] = sum_exp_t / np.tile(np.reshape(lambda_t[t,:],(1,1,Num_month_train)),(Num_st_Oi,t,1))
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
            Pn_id_cir[l,t,:] = np.log(scipy.stats.poisson.pmf(n_idlt, Lambda_t))
            #
        num_00 = np.argwhere(lambda_t == 0)
        Pn_id_cir[l,num_00[:,0],num_00[:,1]] = 0     
    Pn_id_cir_sum = Pn_id_cir.sum()
    #
    log_Pgamma_i = Pn_id_cir_sum
    return log_Pgamma_i, Alpha_i





def log_gamma_i_z(gamma_i,Theta,Z,i):
    import numpy as np
    r = np.argwhere(Z[i,:] ==1)[0,0]  
    Pgamma_i_cir = np.zeros((Num_st,R))
    for j in range(0,Num_st): #station
        for r1 in range(0,R):
            gamma_ij = gamma_i[j]
            theta_rr1 = Theta[r,r1]
            if theta_rr1 == 0:
                theta_rr1 = theta_rr1 + math.exp(-1)   
            if theta_rr1 == 1:
                theta_rr1 = theta_rr1 - math.exp(-1)  
            Bernoulli = np.log(theta_rr1) * gamma_ij +np.log (1-theta_rr1) * (1-gamma_ij)
            Pgamma_i_cir[j,r1] = Bernoulli*Z[j,r1]
    Pgamma_i_cir_sum = (Pgamma_i_cir).sum() 
    return Pgamma_i_cir_sum







def model_proposed_test(flow_test,flow_test_CIF,All_Mu,All_Alpha,R):
    import numpy as np
    Num_st = flow_train.shape[1] #nodes
    Num_time = flow_train.shape[0] #time points
    Num_month_train = flow_train.shape[2] #samples
    Num_month_test = flow_test.shape[2] #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = st_info.shape[1] #attributes
    L = flow_train.shape[3] #layers
    #
    Nend = (np.argwhere((All_Mu.sum(axis=0).sum(axis=0).sum(axis=0)==0))[:,0]).min()-1
    Mu = All_Mu[:,:,:,Nend]
    Alpha = All_Alpha[:,:,:,Nend] 
    Num_day_test = flow_test.shape[2]
    hat_flow = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
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
            hat_flow[:,i,:,l] = lambda_t
    #            
    #
    MAE = abs(hat_flow - flow_test).mean() 
    RMAE = abs(hat_flow - flow_test).sum() / flow_test.sum() 
    MAE_CIF = abs(hat_flow - flow_test_CIF).mean() 
    RMAE_CIF = abs(hat_flow - flow_test_CIF).sum() / flow_test_CIF.sum()
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF



######################################################################
#Main function
Num_st = 25 #Set the number of nodes to 5,10,15,25, respectively.
R = 3 #communities
#True parameters


#Data generation
st_info, flow_train, flow_train_CIF, flow_test, flow_test_CIF = data_generating(Num_st,R)

###################################
#Parameter initialization

#community number
km = KMeans(n_clusters=R)
km.fit(st_info)
y_pred1 = km.labels_
tal0 = 1/int(R/2)
Tal = (1-tal0)/(R-1) * np.ones((Num_st,R))
for i in range(0,Num_st):
    Tal[i,y_pred1[i]] = tal0


com = np.zeros((Num_st))
for i in range(0,Num_st): #station
    com[i] = np.argwhere(Tal[i,:] == Tal[i,:].max())[0,0]


myfile = open('community number_attributes.csv','w')
with myfile:
    writer = csv.writer(myfile)
    writer.writerows(np.reshape(com,(com.shape[0],1)))



#Figure
#attributes data
sns.heatmap(st_info,linewidths=.1,cmap="YlGnBu")
pyplot.show()

#count data
day = 0

#CIF
pyplot.subplot(3,1,1)
pyplot.plot(flow_train_CIF[:,:,day,0])
pyplot.subplot(3,1,2)
pyplot.plot(flow_train_CIF[:,:,day,1])
pyplot.subplot(3,1,3)
pyplot.plot(flow_train_CIF[:,:,day,2])
pyplot.show()

#count
pyplot.subplot(3,1,1)
pyplot.plot(flow_train[:,:,day,0])
pyplot.subplot(3,1,2)
pyplot.plot(flow_train[:,:,day,1])
pyplot.subplot(3,1,3)
pyplot.plot(flow_train[:,:,day,2])
pyplot.show() 



Num_time = 100 #time points
Num_month_train = 50 #samples
Num_month_test = 50 #samples
t_T = Num_time #last time
t_1 = 0 #initial time
K = 18 #attributes
L = 3 #layers
delta = 0.2


Np = 100 # repeated times
#Parameters
All = 11
Proposed_Theta = np.ones((R,R,All,Np))
Proposed_Phi = np.ones((R,K,All,Np))
Proposed_Tal = np.zeros((Num_st,R,All,Np))
Proposed_Z = np.zeros((Num_st,R,All,Np))
Proposed_Mu = np.zeros((Num_st,t_T-t_1,L,All,Np))
Proposed_Alpha = np.zeros((Num_st,Num_st,L,All,Np))
Proposed_Gamma = np.zeros((Num_st,Num_st,All,Np))
#Data
Proposed_flow_train = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Proposed_flow_train_CIF = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Proposed_flow_test = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Proposed_flow_test_CIF = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Proposed_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Proposed_st_info = np.zeros((Num_st,K,Np))
#Metrics
Proposed_metrics = np.zeros((5,Np)) #MAE, RMAE, Training time

for ip in range(0,Np):
    ip
    #Generate data
    st_info, flow_train, flow_train_CIF, flow_test, flow_test_CIF = data_generating(Num_st,R)
    #Model
    All_Theta,All_Phi,All_Tal,All_Z,All_Mu,All_Alpha,All_Gamma,Time = model_proposed(flow_train,st_info,R)
    #
    #model testing
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = model_proposed_test(flow_test,flow_test_CIF,All_Mu,All_Alpha,R)
    #
    print('MAE = ', MAE) 
    print('RMAE = ', RMAE) 
    print('MAE_CIF = ', MAE_CIF) 
    print('RMAE_CIF = ', RMAE_CIF)
    #
    Proposed_Theta[:,:,:,ip] = All_Theta
    Proposed_Phi[:,:,:,ip] = All_Phi
    Proposed_Tal[:,:,:,ip] = All_Tal
    Proposed_Z[:,:,:,ip] = All_Z
    Proposed_Mu[:,:,:,:,ip] = All_Mu
    Proposed_Alpha[:,:,:,:,ip] = All_Alpha
    Proposed_Gamma[:,:,:,ip] = All_Gamma
    #
    Proposed_st_info[:,:,ip] = st_info
    Proposed_flow_train[:,:,:,:,ip] = flow_train
    Proposed_flow_train_CIF[:,:,:,:,ip] = flow_train_CIF
    Proposed_flow_test[:,:,:,:,ip] = flow_test
    Proposed_flow_test_CIF[:,:,:,:,ip] = flow_test_CIF
    Proposed_hat_flow[:,:,:,:,ip] = hat_flow
    #
    Proposed_metrics[0,ip] = MAE
    Proposed_metrics[1,ip] = RMAE
    Proposed_metrics[2,ip] = Time
    Proposed_metrics[3,ip] = MAE_CIF
    Proposed_metrics[4,ip] = RMAE_CIF



    
#
np.save('Proposed_Theta.npy',Proposed_Theta)
np.save('Proposed_Phi.npy',Proposed_Phi)
np.save('Proposed_Tal.npy',Proposed_Tal)
np.save('Proposed_Z.npy',Proposed_Z)
np.save('Proposed_Mu.npy',Proposed_Mu)
np.save('Proposed_Alpha.npy',Proposed_Alpha)
np.save('Proposed_Gamma.npy',Proposed_Gamma)
np.save('Proposed_flow_train.npy',Proposed_flow_train)
np.save('Proposed_flow_train_CIF.npy',Proposed_flow_train_CIF)
np.save('Proposed_flow_test.npy',Proposed_flow_test)
np.save('Proposed_flow_test_CIF.npy',Proposed_flow_test_CIF)
np.save('Proposed_hat_flow.npy',Proposed_hat_flow)
np.save('Proposed_metrics.npy',Proposed_metrics)
np.save('Proposed_st_info.npy',Proposed_st_info)   



#######################################################
#Table B2: Metircs
Proposed_metrics = np.load('Proposed_metrics.npy')

Np0 = Np
all_MAEmean = Proposed_metrics[0,:Np0].mean()
print('MAE mean = ',all_MAEmean)


all_MAEstd = Proposed_metrics[0,:Np0].std()
print('MAE std = ',all_MAEstd) 


all_RMAEmean = Proposed_metrics[1,:Np0].mean()
print('RMAE mean = ',all_RMAEmean) 


all_RMAEstd = Proposed_metrics[1,:Np0].std()
print('RMAE std = ',all_RMAEstd) 


all_MAE_CIFmean = Proposed_metrics[3,:Np0].mean()
print('MAE mean = ',all_MAE_CIFmean) 


all_MAE_CIFstd = Proposed_metrics[3,:Np0].std() 
print('MAE std = ',all_MAE_CIFstd) 


all_RMAE_CIFmean = Proposed_metrics[4,:Np0].mean()
print('MAE mean = ',all_RMAE_CIFmean) 


all_RMAE_CIFstd = Proposed_metrics[4,:Np0].std()
print('MAE std = ',all_RMAE_CIFstd) 


all_Timemean = Proposed_metrics[2,:Np0].mean()
print('MAE mean = ',all_Timemean) 


all_Timestd = Proposed_metrics[2,:Np0].std()
print('MAE std = ',all_Timestd) 

