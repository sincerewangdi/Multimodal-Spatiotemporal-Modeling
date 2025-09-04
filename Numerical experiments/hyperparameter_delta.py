#Proposed method with various bandwidth values \delta

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


Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
Hp = 10 # repeated times


delta_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Hp))
delta_metrics = np.zeros((6,Hp))
for ip in range(0,Hp):
    ip
    delta = (ip+1) * 0.05
    flow_train = Proposed_flow_train[:,:,:,:,0] #We randomly choose one sample, e.g., sample 0
    flow_test = Proposed_flow_test[:,:,:,:,0]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,0]
    st_info = Proposed_st_info[:,:,0]
    All_Theta,All_Phi,All_Tal,All_Z,All_Mu,All_Alpha,All_Gamma,Time = model_proposed_delta(flow_train,st_info,delta)
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = model_proposed_test(flow_test,flow_test_CIF,All_Mu,All_Alpha,delta)
    print('MAE = ', MAE)
    print('RMAE = ', RMAE)
    print('MAE_CIF = ', MAE_CIF)
    print('RMAE_CIF = ', RMAE_CIF)
    delta_hat_flow[:,:,:,:,ip] = hat_flow
    delta_metrics[0,ip] = MAE
    delta_metrics[1,ip] = RMAE
    delta_metrics[2,ip] = Time
    delta_metrics[3,ip] = MAE_CIF
    delta_metrics[4,ip] = RMAE_CIF
    delta_metrics[5,ip] = delta


np.save('delta_hat_flow.npy',delta_hat_flow)
np.save('delta_metrics.npy',delta_metrics)



#Figure 5: MAEs of CIFs with various bandwidth values
delta_metrics = np.load('delta_metrics.npy')
delta_hat_flow = np.load('delta_hat_flow.npy')
pyplot.figure(figsize=(5, 4))
pyplot.plot(delta_metrics[5,:], delta_metrics[3,:], 'bo--', label='count')
pyplot.xlabel('$\delta$')
pyplot.ylabel('MAE')
pyplot.grid(linestyle = ':')
pyplot.show()







