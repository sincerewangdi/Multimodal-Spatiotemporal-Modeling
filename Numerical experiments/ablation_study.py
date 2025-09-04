# Ablation study
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




Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')

Np = 100 # repeated times



#######################################################################
#1 Neither attributes nor communities
noattrnocomv2_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
noattrnocomv2_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    st_info = Proposed_st_info[:,:,ip]
    All_Mu,All_Alpha,Time= Ablation_noattrnocomv2(flow_train,st_info)
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = model_proposed_test1(flow_test,flow_test_CIF,All_Mu,All_Alpha)
    print('MAE = ', MAE) #1.8180984153816853
    print('RMAE = ', RMAE) #0.03484726740268056
    print('MAE_CIF = ', MAE_CIF) #0.11625595107361368
    print('RMAE_CIF = ', RMAE_CIF) #0.0022284567073599148
    noattrnocomv2_hat_flow[:,:,:,:,ip] = hat_flow
    noattrnocomv2_metrics[0,ip] = MAE
    noattrnocomv2_metrics[1,ip] = RMAE
    noattrnocomv2_metrics[2,ip] = Time
    noattrnocomv2_metrics[3,ip] = MAE_CIF
    noattrnocomv2_metrics[4,ip] = RMAE_CIF


np.save('noattrnocomv2_hat_flow.npy',noattrnocomv2_hat_flow)
np.save('noattrnocomv2_metrics.npy',noattrnocomv2_metrics)




#####################################################################
#2 Communities learnt by attributes
attrnocomv2_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
attrnocomv2_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    st_info = Proposed_st_info[:,:,ip]
    All_Mu,All_Gamma,Time= Ablation_attrnocomv2(flow_train,st_info)
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = model_proposed_test2(flow_test,flow_test_CIF,All_Mu,All_Alpha)
    print('MAE = ', MAE) #1.8261472926856912
    print('RMAE = ', RMAE) #0.035001539238204475
    print('MAE_CIF = ', MAE_CIF) #0.20485108785297151
    print('RMAE_CIF = ', RMAE_CIF) #0.003926696023043766
    attrnocomv2_hat_flow[:,:,:,:,ip] = hat_flow
    attrnocomv2_metrics[0,ip] = MAE
    attrnocomv2_metrics[1,ip] = RMAE
    attrnocomv2_metrics[2,ip] = Time
    attrnocomv2_metrics[3,ip] = MAE_CIF
    attrnocomv2_metrics[4,ip] = RMAE_CIF


np.save('attrnocomv2_hat_flow.npy',attrnocomv2_hat_flow)
np.save('attrnocomv2_metrics.npy',attrnocomv2_metrics)




#####################################################################
#3 Communities learnt by event frequencies
noattr_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
noattr_metrics = np.zeros((5,Np))
for ip in range(1,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    st_info = Proposed_st_info[:,:,ip]
    All_Mu,All_Alpha,All_Gamma,Time= Ablation_noattr(flow_train,st_info)
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = model_proposed_test3(flow_test,flow_test_CIF,All_Mu,All_Alpha)
    print('MAE = ', MAE) #1.817988974423656
    print('RMAE = ', RMAE) #0.03484516976137745
    print('MAE_CIF = ', MAE_CIF) #0.11350550651668069
    print('RMAE_CIF = ', RMAE_CIF) #0.0021757347041891878
    noattr_hat_flow[:,:,:,:,ip] = hat_flow
    noattr_metrics[0,ip] = MAE
    noattr_metrics[1,ip] = RMAE
    noattr_metrics[2,ip] = Time
    noattr_metrics[3,ip] = MAE_CIF
    noattr_metrics[4,ip] = RMAE_CIF


np.save('noattr_hat_flow.npy',noattr_hat_flow)
np.save('noattr_metrics.npy',noattr_metrics)






#######################################################################
#Table 3: Metrics
noattr_metrics = np.load('noattr_metrics.npy') #3 Communities learnt by event frequencies
noattrnocomv2_metrics = np.load('noattrnocomv2_metrics.npy') #2 Communities learnt by attributes
attrnocomv2_metrics = np.load('attrnocomv2_metrics.npy') #1 Neither attributes nor communities


Np0 = Np
all_MAEmean = np.zeros((3,))
all_MAEmean[2] = noattr_metrics[0,:Np0].mean()
all_MAEmean[1] = attrnocomv2_metrics[0,:Np0].mean()
all_MAEmean[0] = noattrnocomv2_metrics[0,:Np0].mean()
print('MAE mean = ',all_MAEmean)
#MAE mean =  [1.81982426 1.83793239 1.81946004]


all_MAEstd = np.zeros((3,))
all_MAEstd[2] = noattr_metrics[0,:Np0].std()
all_MAEstd[1] = attrnocomv2_metrics[0,:Np0].std()
all_MAEstd[0] = noattrnocomv2_metrics[0,:Np0].std()
print('MAE std = ',all_MAEstd)
#MAE std =  [0.00145084 0.0017894  0.00143492]


all_RMAEmean = np.zeros((3,))
all_RMAEmean[2] = noattr_metrics[1,:Np0].mean()
all_RMAEmean[1] = attrnocomv2_metrics[1,:Np0].mean()
all_RMAEmean[0] = noattrnocomv2_metrics[1,:Np0].mean()
print('RMAE mean = ',all_RMAEmean)
#RMAE mean =  [0.03488885 0.03523601 0.03488187]


all_RMAEstd = np.zeros((3,))
all_RMAEstd[2] = noattr_metrics[1,:Np0].std()
all_RMAEstd[1] = attrnocomv2_metrics[1,:Np0].std()
all_RMAEstd[0] = noattrnocomv2_metrics[1,:Np0].std()
print('RMAE std = ',all_RMAEstd)
#RMAE std =  [3.57648859e-05 4.10092493e-05 3.53980828e-05]


all_MAE_CIFmean = np.zeros((3,))
all_MAE_CIFmean[2] = noattr_metrics[3,:Np0].mean()
all_MAE_CIFmean[1] = attrnocomv2_metrics[3,:Np0].mean()
all_MAE_CIFmean[0] = noattrnocomv2_metrics[3,:Np0].mean()
print('MAE mean = ',all_MAE_CIFmean)
#MAE mean =  [0.11527788 0.26563405 0.11095428]


all_MAE_CIFstd = np.zeros((3,))
all_MAE_CIFstd[2] = noattr_metrics[3,:Np0].std()
all_MAE_CIFstd[1] = attrnocomv2_metrics[3,:Np0].std()
all_MAE_CIFstd[0] = noattrnocomv2_metrics[3,:Np0].std()
print('MAE std = ',all_MAE_CIFstd)
#MAE std =  [0.00228432 0.0038824  0.00225647]


all_RMAE_CIFmean = np.zeros((3,))
all_RMAE_CIFmean[2] = noattr_metrics[4,:Np0].mean()
all_RMAE_CIFmean[1] = attrnocomv2_metrics[4,:Np0].mean()
all_RMAE_CIFmean[0] = noattrnocomv2_metrics[4,:Np0].mean()
print('MAE mean = ',all_RMAE_CIFmean)
#MAE mean =  [0.00221006 0.00509261 0.00212717]


all_RMAE_CIFstd = np.zeros((3,))
all_RMAE_CIFstd[2] = noattr_metrics[4,:Np0].std()
all_RMAE_CIFstd[1] = attrnocomv2_metrics[4,:Np0].std()
all_RMAE_CIFstd[0] = noattrnocomv2_metrics[4,:Np0].std()
print('MAE std = ',all_RMAE_CIFstd)
#MAE std =  [4.39066565e-05 7.41934428e-05 4.33812468e-05]


all_Timemean = np.zeros((3,))
all_Timemean[2] = noattr_metrics[2,:Np0].mean()
all_Timemean[1] = attrnocomv2_metrics[2,:Np0].mean()
all_Timemean[0] = noattrnocomv2_metrics[2,:Np0].mean()
print('MAE mean = ',all_Timemean)
#MAE mean =  [3022.6  535.9 2812.8]


