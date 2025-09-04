def data_generating():
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
    Num_st = 20 #nodes
    Num_time = 100 #time points
    Num_month_train = 50 #samples
    Num_month_test = 50 #samples
    t_T = Num_time #last time
    t_1 = 0 #initial time
    K = 18 #attributes
    L = 3 #layers
    R = 3 #communities
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


