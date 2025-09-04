######################################################################
#1 Neither attributes nor communities
def Ablation_noattrnocomv2(flow_train,st_info):
    starttime = datetime.datetime.now()
    #
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
    #############################################
    #Parameter initialization
    #
    #
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
            Oi = np.array([i for i in range(0,Num_st)])
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
    All = 200 # total iteration number
    All_Mu = np.zeros((Num_st,t_T-t_1,L,All))
    All_Alpha = np.zeros((Num_st,Num_st,L,All))
    #
    All_Mu[:,:,:,0] = Mu
    All_Alpha[:,:,:,0] = Alpha
    #
    #
    #
    for numa in range(0,All-1):  
        #numa
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
            for i in range(0,Num_st): 
                Oi = np.array([i for i in range(0,Num_st)])
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
        Q = 2 #iterative number
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
    #
    #
    #
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    #
    return All_Mu,All_Alpha,Time






def model_proposed_test1(flow_test,flow_test_CIF,All_Mu,All_Alpha):
    import numpy as np
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
    Nend = (np.argwhere((All_Mu.sum(axis=0).sum(axis=0).sum(axis=0)==0))[:,0]).min()-1
    Mu = All_Mu[:,:,:,Nend]
    Alpha = All_Alpha[:,:,:,Nend] 
    Num_day_test = flow_test.shape[2]
    hat_flow = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
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
            hat_flow[:,i,:,l] = lambda_t
    #            
    #
    MAE = abs(hat_flow - flow_test).mean() 
    RMAE = abs(hat_flow - flow_test).sum() / flow_test.sum() 
    MAE_CIF = abs(hat_flow - flow_test_CIF).mean() 
    RMAE_CIF = abs(hat_flow - flow_test_CIF).sum() / flow_test_CIF.sum()
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF





######################################################################
#2 Communities learnt by attributes
def Ablation_attrnocomv2(flow_train,st_info):
    starttime = datetime.datetime.now()
    #
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
    #############################################
    #Parameter initialization
    #Gamma is given 
    Gamma = np.zeros((Num_st,Num_st,L))
    for num_c in range(0,R):     
        orignal_com = pandas.read_csv('community number_attributes.csv',header=None) #station information
        com = orignal_com.values
        Num_c = np.argwhere(com==num_c)[:,0]
        for ic1 in range(0,Num_c.shape[0]):
            for ic2 in range(0,Num_c.shape[0]):
                Gamma[Num_c[ic1],Num_c[ic2],:] = 1
    #
    #
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
            Oi = np.argwhere((Gamma[i,:,l]==1))[:,0]
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
    All = 2 # total iteration number
    All_Mu = np.zeros((Num_st,t_T-t_1,L,All))
    All_Alpha = np.zeros((Num_st,Num_st,L,All))
    #
    All_Mu[:,:,:,0] = Mu
    All_Alpha[:,:,:,0] = Alpha
    #
    #
    #
    for numa in range(0,All-1):  
        #numa
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
                Oi = np.argwhere((Gamma[i,:,l]==1))[:,0]
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
        Q = 200 #iterative number
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
        All_Mu[:,:,:,numa] = Mu
        All_Alpha[:,:,:,numa] = Alpha
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    #
    return All_Mu,All_Alpha,Time







def model_proposed_test2(flow_test,flow_test_CIF,All_Mu,All_Alpha):
    import numpy as np
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
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
#3 Communities learnt by event frequencies
def Ablation_noattr(flow_train,st_info):
    starttime = datetime.datetime.now()
    #
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
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
    #Parameter estimation
    All = 11 # total iteration number
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
        '''
        sns.heatmap(Alpha[:,:,l],linewidths=.1,cmap="YlGnBu")
        pyplot.show()
        sns.heatmap(Gamma,linewidths=.1,cmap="YlGnBu")
        pyplot.show()
        '''
        #    
        #
        #
        All_Mu[:,:,:,numa] = Mu
        All_Alpha[:,:,:,numa] = Alpha
        #All_Gamma[:,:,numa] = Gamma
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
                #Pgamma_i_cir_sum[j] = log_gamma_i_z(gamma_i,Theta,Z,i)
                Log_posterior[j] = Log_Pgamma[j] #+ Pgamma_i_cir_sum[j] 
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
    return All_Mu,All_Alpha,All_Gamma,Time










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







def model_proposed_test3(flow_test,flow_test_CIF,All_Mu,All_Alpha):
    import numpy as np
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
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




