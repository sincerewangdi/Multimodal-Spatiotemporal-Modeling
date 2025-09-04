######################################################################
# benchmark: S-Hawkes
def ST_Hawkes(flow_train,st_info):
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
    #########################
    #Parameter initialization
    #Mu, Alpha
    Num_time_pre = flow_train.shape[0]
    Num_st = flow_train.shape[1]
    Num_month_train = flow_train.shape[2]
    L = flow_train.shape[3]
    t_T = Num_time_pre
    t_1 = 0
    delta = 0.2
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
    Alpha = np.zeros((Num_st,Num_st,L)) 
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
    All_Mu[:,:,:,0] = Mu
    All_Alpha[:,:,:,0] = Alpha
    Alpha = All_Alpha[:,:,:,0]
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
    All_Mu[:,:,:,numa] = Mu
    All_Alpha[:,:,:,numa] = Alpha
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    #
    return All_Mu,All_Alpha,Time




def ST_Hawkes_test(flow_test,flow_test_CIF,All_Mu,All_Alpha):
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
    Nend = 0
    delta = 0.2
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
                n_jdltj = np.ones((t, Num_st, Num_month_test)) #flow_test[:t,:,:,l] 
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
# benchmark: Multimodal S-Hawkes
def multimodalST_Hawkes(flow_train,st_info):
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
    delta = 0.2
    ##########################
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




def multimodalST_Hawkes_test(flow_test,flow_test_CIF,All_Mu,All_Alpha,Phi):
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
    Nend = 0
    Mu = All_Mu[:,:,:,Nend]
    Alpha = All_Alpha[:,:,:,Nend] 
    Num_day_test = flow_test.shape[2]
    hat_flow = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],L))
    Num_month_test = flow_test.shape[2]
    delta = 0.2
    for l in range(0,L):
        #l = 0
        for i in range(0,Num_st): 
            lambda_t = np.zeros((t_T-t_1,Num_month_test))
            #
            Px_id_cir = np.zeros((K,))
            for k in range(0,K):
                x_ik = st_info[i,k]
                Px_id_cir[k] = poisson.pmf(x_ik, Phi[k])   
            Px_id_cir_sum = np.prod(Px_id_cir)
            #         
            t = 0
            lambda_t[t,:] = Mu[i,t,l] * Px_id_cir_sum
            for t in range(t_1+1, t_T):
                n_jdltj = np.ones((t, Num_st, Num_month_test))   
                theta = Alpha[i,:,l]
                exp = np.array([np.exp( - (t - t_j) / delta) for t_j in range(t_1, t)]) 
                sum_exp_t = np.tile(np.reshape(theta,(Num_st,1,1)),(1,t,Num_month_test)) \
                            * np.tile(np.reshape(exp,(1,t,1)),(Num_st,1,Num_month_test)) * np.transpose(n_jdltj,(1,0,2))
                lambda_t[t,:] = (Mu[i,t,l] + sum_exp_t.sum(axis = 0).sum(axis = 0)) * Px_id_cir_sum
            hat_flow[:,i,:,l] = lambda_t
    #            
    #
    MAE = abs(hat_flow - flow_test).mean() 
    RMAE = abs(hat_flow - flow_test).sum() / flow_test.sum() 
    MAE_CIF = abs(hat_flow - flow_test_CIF).mean() 
    RMAE_CIF = abs(hat_flow - flow_test_CIF).sum() / flow_test_CIF.sum()
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF




###################################################
#Benchmark: T-Hawkes
#Hawkes process that varies with time for each node respectivelye

def model_Thawkes(flow_train,st_info,flow_test,flow_test_CIF):
    starttime = datetime.datetime.now()
    #
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    L = 3 #layers
    R = 3 #communities
    #model training
    Lambda0 = np.zeros((Num_st, L))
    Alpha = np.zeros((Num_st, L))
    Beta = np.zeros((Num_st, L))
    for l in range(0,L):
        for st in range(0,Num_st):
            #EM Algorithm
            Month = Num_month_train
            T = Num_time
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
    #
    #
    #
    #model testing
    Num_day_test = flow_test.shape[2]
    hat_flow = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
    for l in range(0,L):
        for st in range(0,Num_st):
            #print(st)
            demand = flow_test[:,st,:,l]
            lambda0 = Lambda0[st,l]
            beta = Beta[st,l]
            alpha = Alpha[st,l]
            for day in range(0,Num_day_test):           
                demand_day = demand[:,day]
                Lambda = np.zeros((T,))
                for it in range(0, T):
                    g = np.zeros((T,T))
                    for jt in range(0,it):
                        g[it,jt] = np.exp(-beta * (it - jt)) * demand_day[jt]
                    Lambda[it] = lambda0 + alpha * g[it,:].sum()
                hat_flow[:,st,day,l] = Lambda
    #
    #
    MAE = abs(hat_flow - flow_test).mean() 
    RMAE = abs(hat_flow - flow_test).sum() / flow_test.sum() 
    MAE_CIF = abs(hat_flow - flow_test_CIF).mean() 
    RMAE_CIF = abs(hat_flow - flow_test_CIF).sum() / flow_test_CIF.sum()
    #
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time




########################################
#Benchmark: ConvLSTM
def model_ConvLSTM(flow_train,st_info,flow_test,flow_test_CIF,p):
    starttime = datetime.datetime.now()
    #model training
    hat_flow = np.zeros((Num_month_train,t_T-t_1-p,Num_st,L))
    test0 = np.zeros((Num_month_train,t_T-t_1-p,Num_st,L))
    test0_CIF = np.zeros((Num_month_train,t_T-t_1-p,Num_st,L))
    for l in range(0,L):
        l
        trainX = np.zeros((Num_month_train,t_T-t_1-p,p * Num_st))
        trainY = np.zeros((Num_month_train,t_T-t_1-p,Num_st))
        testX = np.zeros((Num_month_train,t_T-t_1-p,p * Num_st))
        testY = np.zeros((Num_month_train,t_T-t_1-p,Num_st))
        testCIF = np.zeros((Num_month_train,t_T-t_1-p,Num_st))
        for i in range(0,Num_st):
            traindata = flow_train[:,i,:,l].T
            testdata = flow_test[:,i,:,l].T
            testdata_CIF = flow_test_CIF[:,i,:,l].T
            for ip in range(0,t_T-t_1-p):
                trainX[:,ip,i * p : (i+1) * p] = traindata[:,ip:ip+p]
                trainY[:,ip,i] = traindata[:,ip+p]
                testX[:,ip,i * p : (i+1) * p] = testdata[:,ip:ip+p]
                testY[:,ip,i] = testdata[:,ip+p]
                testCIF[:,ip,i] = testdata_CIF[:,ip+p]
        #
        #
        #lstm
        Nh = 2000
        lstm = Sequential()
        lstm.add(LSTM(Nh, input_shape=(trainX.shape[1], trainX.shape[2]), dropout=0.5, return_sequences=True)) 
        lstm.add(Dense(Num_st, activation='linear'))
        lstm.summary()
        lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        history = lstm.fit(trainX, trainY, epochs=20, batch_size=10)
        #model testing
        predY = lstm.predict(testX)
        test0[:,:,:,l] = testY
        test0_CIF[:,:,:,l] = testCIF
        hat_flow[:,:,:,l] = predY
    #
    #
    MAE = abs(hat_flow - test0).mean()
    RMAE = abs(hat_flow - test0).sum() / test0.sum()
    MAE_CIF = abs(hat_flow - test0_CIF).mean()
    RMAE_CIF = abs(hat_flow - test0_CIF).sum() / test0_CIF.sum() 
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time



########################################
#Benchmark: STCM
def model_logGauss(flow_train,st_info,flow_test,flow_test_CIF,p):
    starttime = datetime.datetime.now()
    #
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    L = 3 #layers
    R = 3 #communities
    ###################################  
    hat_flow = np.zeros((Num_st * (t_T-t_1-p), Num_month_train,L))
    test0 = np.zeros((Num_st * (t_T-t_1-p), Num_month_train,L))
    test0_CIF = np.zeros((Num_st * (t_T-t_1-p), Num_month_train,L))
    #
    for l in range(0,L):
        l
        Lambda = flow_train[:,:,:,l].mean(axis = 2)
        U = np.log(Lambda) 
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
        testU = np.log(flow_test[:,:,:,l])
        pred_TestU = np.zeros((Num_st * (t_T-t_1-p), Num_month_train))
        for day in range(0,Num_month_train):
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
            pred_TestU[:,day] = np.array(pred_testU_day)[:,0]
        #
        #
        #
        testU = flow_test
        TestU = np.zeros((Num_st * (t_T-t_1-p), Num_month_train))
        for day in range(0,Num_month_train):
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
        testU = flow_test_CIF
        TestU_CIF = np.zeros((Num_st * (t_T-t_1-p), Num_month_train))
        for day in range(0,Num_month_train):
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
            TestU_CIF[:,day] = np.array(U1)[:,0]
        #
        #
        hat_flow[:,:,l] = pred_TestU
        test0[:,:,l] = TestU
        test0_CIF[:,:,l] = TestU_CIF
        #
    MAE = abs(hat_flow - test0).mean()
    RMAE = abs(hat_flow - test0).sum() / test0.sum()
    MAE_CIF = abs(hat_flow - test0_CIF).mean()
    RMAE_CIF = abs(hat_flow - test0_CIF).sum() / test0_CIF.sum() 
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time







############################################################################
# Benchmark: MSHP
def model_Chawkes(flow_train,st_info,flow_test,flow_test_CIF):
    starttime = datetime.datetime.now()
    #
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    L = 3 #layers
    R = 3 #communities
    ###################################
    #Parameter initialization
    #Tal
    #Attributes only to classify community
    km = KMeans(n_clusters=R)
    km.fit(st_info)
    y_pred1 = km.labels_
    tal0 = 1/int(R/2)
    Tal = (1-tal0)/(R-1) * np.ones((Num_st,R))
    for i in range(0,Num_st):
        Tal[i,y_pred1[i]] = tal0
    #
    #
    #Phi
    Q = 20 #iterative number
    Phi = np.ones((R,K))
    Pidr = np.zeros((Num_st,R))
    Q_Phi = np.ones((R,K,Q))
    Q_Tal = np.zeros((Num_st,R,Q))
    for q in range(0,Q):
        for r in range(0,R): 
            for i in range(0,Num_st): 
                #d = 0
                #r = 0
                #i = 0
                #
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
        #
        #  
    Q = 10 #iterative number
    Q_Mu = 1 * np.zeros((Num_st,t_T-t_1,L,Q))
    Q_Theta = np.zeros((R,Num_st,L,Q))
    Q_Tal = np.zeros((Num_st,R,Q))
    Q_Phi = np.ones((R,K,Q))
    Mu = 0.1 * np.ones((Num_st,t_T-t_1,L))
    Theta = np.ones((R,Num_st,L)) * 1e-2
    Phi = Phi
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
        for r in range(0,R):  
            #r
            #       
            for i in range(0,Num_st): 
                #i
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
        hat_tal_ird = np.zeros((Num_st,R,Num_month_train))
        for i in range(0,Num_st): 
            #print(i)
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
        # M step
        sum_tal0 = hat_tal_ird.sum(axis = 2)
        sum_tal1 = hat_tal_ird.sum(axis = 2).sum(axis = 1)
        #
        #Tal
        Tal =  sum_tal0 / np.tile(np.reshape(sum_tal1,(Num_st,1)),(1,R))
        Q_Tal[:,:,q] = Tal
        #
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
    #
    #
    #model testing
    Num_day_test = flow_test.shape[2]
    hat_flow = np.zeros((flow_test.shape[0],flow_test.shape[1],flow_test.shape[2],flow_test.shape[3]))
    Num_month_test = flow_test.shape[2]
    for i in range(0,Num_st): 
        r = np.argwhere(Tal[i,:] ==Tal[i,:].max())[0,0]
        for l in range(0,L):
            lambda_t = np.zeros((t_T-t_1,Num_month_test))
            t = 1
            lambda_t[t,:] = Mu[i,t,l]
            for t in range(t_1+1, t_T):
                n_jdltj = flow_test[:t,:,:,l]
                theta = Theta[r,:,l]
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
    #
    endtime = datetime.datetime.now()
    Time = (endtime - starttime).seconds
    return hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time












