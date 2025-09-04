# Model comparison
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences




Num_st = 20 #nodes
Num_time = 100 #time points
Num_month_train = 50 #samples
Num_month_test = 50 #samples
t_T = Num_time #last time
t_1 = 0 #initial time
K = 18 #attributes
L = 3 #layers
R = 3 #communities


#######################################################################
#benchmark: S-Hawkes
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')

Np = 100 # repeated times
ST_Hawkes_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
ST_Hawkes_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    All_Mu,All_Alpha,Time= ST_Hawkes(flow_train,st_info)
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = ST_Hawkes_test(flow_test,flow_test_CIF,All_Mu,All_Alpha)
    print('MAE = ', MAE) #2.3730767211739736
    print('RMAE = ', RMAE) #0.045484467930997634
    print('MAE_CIF = ', MAE_CIF) #1.1891016298479846
    print('RMAE_CIF = ', RMAE_CIF) #0.022793340713280527
    ST_Hawkes_hat_flow[:,:,:,:,ip] = hat_flow
    ST_Hawkes_metrics[0,ip] = MAE
    ST_Hawkes_metrics[1,ip] = RMAE
    ST_Hawkes_metrics[2,ip] = Time
    ST_Hawkes_metrics[3,ip] = MAE_CIF
    ST_Hawkes_metrics[4,ip] = RMAE_CIF


np.save('ST_Hawkes_hat_flow.npy',ST_Hawkes_hat_flow)
np.save('ST_Hawkes_metrics.npy',ST_Hawkes_metrics)


################################################################
#benchmark: multimodal S-Hawkes
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
#st_info = np.load('st_info.npy')
Np = 100 # repeated times

multimodalST_Hawkes_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
multimodalST_Hawkes_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    st_info = Proposed_st_info[:,:,ip]
    All_Mu,All_Alpha,Phi,Time= multimodalST_Hawkes(flow_train,st_info)
    hat_flow, MAE, RMAE, MAE_CIF, RMAE_CIF = multimodalST_Hawkes_test(flow_test,flow_test_CIF,All_Mu,All_Alpha,Phi)
    print('MAE = ', MAE) #2.3730789633993363
    print('RMAE = ', RMAE) #0.04548451090745376
    print('MAE_CIF = ', MAE_CIF) #1.1891049384465515
    print('RMAE_CIF = ', RMAE_CIF) #0.022793404134280486
    multimodalST_Hawkes_hat_flow[:,:,:,:,ip] = hat_flow
    multimodalST_Hawkes_metrics[0,ip] = MAE
    multimodalST_Hawkes_metrics[1,ip] = RMAE
    multimodalST_Hawkes_metrics[2,ip] = Time
    multimodalST_Hawkes_metrics[3,ip] = MAE_CIF
    multimodalST_Hawkes_metrics[4,ip] = RMAE_CIF


np.save('multimodalST_Hawkes_hat_flow.npy',multimodalST_Hawkes_hat_flow)
np.save('multimodalST_Hawkes_metrics.npy',multimodalST_Hawkes_metrics)



#################################################################
#T-hawkes

Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
Np = 100 # repeated times

Thawkes_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Thawkes_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    st_info = Proposed_st_info[:,:,ip]
    hat_flow_th,MAE_th,RMAE_th,MAE_CIF_th,RMAE_CIF_th,Time_th = model_Thawkes(flow_train,st_info,flow_test,flow_test_CIF)
    print('MAE = ', MAE_th)
    print('RMAE = ', RMAE_th)
    print('MAE_CIF = ', MAE_CIF_th)
    print('RMAE_CIF = ', RMAE_CIF_th)
    Thawkes_hat_flow[:,:,:,:,ip] = hat_flow_th
    Thawkes_metrics[0,ip] = MAE_th
    Thawkes_metrics[1,ip] = RMAE_th
    Thawkes_metrics[2,ip] = Time_th
    Thawkes_metrics[3,ip] = MAE_CIF_th
    Thawkes_metrics[4,ip] = RMAE_CIF_th


np.save('Thawkes_hat_flow.npy',Thawkes_hat_flow)
np.save('Thawkes_metrics.npy',Thawkes_metrics)




################################################################
#benchmark: ConvLSTM
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
Np = 100 # repeated times

p = 3
ConvLSTM_hat_flow = np.zeros((Num_month_train,t_T-t_1-p,Num_st,L,Np))
ConvLSTM_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time = model_ConvLSTM(flow_train,st_info,flow_test,flow_test_CIF,p)
    print('MAE = ', MAE)
    print('RMAE = ', RMAE)
    print('MAE_CIF = ', MAE_CIF)
    print('RMAE_CIF = ', RMAE_CIF)
    ConvLSTM_hat_flow[:,:,:,:,ip] = hat_flow #(50, 97, 20, 3)
    ConvLSTM_metrics[0,ip] = MAE
    ConvLSTM_metrics[1,ip] = RMAE
    ConvLSTM_metrics[2,ip] = Time
    ConvLSTM_metrics[3,ip] = MAE_CIF
    ConvLSTM_metrics[4,ip] = RMAE_CIF



np.save('ConvLSTM_hat_flow.npy',ConvLSTM_hat_flow)
np.save('ConvLSTM_metrics.npy',ConvLSTM_metrics)




################################################################
#benchmark: STCM
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
Np = 100 # repeated times
Num_time = 100

logGauss_hat_flow = np.zeros(((Num_time-p) * Num_st,Num_month_train,L,Np))
logGauss_metrics = np.zeros((5,Np))
p = 8 #order
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time = model_logGauss(flow_train,st_info,flow_test,flow_test_CIF,p)
    print('MAE = ', MAE)
    print('RMAE = ', RMAE)
    print('MAE_CIF = ', MAE_CIF)
    print('RMAE_CIF = ', RMAE_CIF)
    logGauss_hat_flow[:,:,:,ip] = hat_flow
    logGauss_metrics[0,ip] = MAE
    logGauss_metrics[1,ip] = RMAE
    logGauss_metrics[2,ip] = Time
    logGauss_metrics[3,ip] = MAE_CIF
    logGauss_metrics[4,ip] = RMAE_CIF


np.save('logGauss_hat_flow.npy',logGauss_hat_flow)
np.save('logGauss_metrics.npy',logGauss_metrics)




################################################################
#benchmark: MSHP
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
Np = 100 # repeated times

Chawkes_hat_flow = np.zeros((Num_time,Num_st,Num_month_train,L,Np))
Chawkes_metrics = np.zeros((5,Np))
for ip in range(0,Np):
    ip
    flow_train = Proposed_flow_train[:,:,:,:,ip]
    flow_test = Proposed_flow_test[:,:,:,:,ip]
    flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
    hat_flow,MAE,RMAE,MAE_CIF,RMAE_CIF,Time = model_Chawkes(flow_train,st_info,flow_test,flow_test_CIF)
    print('MAE = ', MAE)
    print('RMAE = ', RMAE)
    print('MAE_CIF = ', MAE_CIF)
    print('RMAE_CIF = ', RMAE_CIF)
    Chawkes_hat_flow[:,:,:,:,ip] = hat_flow
    Chawkes_metrics[0,ip] = MAE
    Chawkes_metrics[1,ip] = RMAE
    Chawkes_metrics[2,ip] = Time
    Chawkes_metrics[3,ip] = MAE_CIF
    Chawkes_metrics[4,ip] = RMAE_CIF


np.save('Chawkes_hat_flow.npy',Chawkes_hat_flow)
np.save('Chawkes_metrics.npy',Chawkes_metrics)





################################################################
################################################################
#Metircs
Thawkes_metrics = np.load('Thawkes_metrics.npy')
ConvLSTM_metrics = np.load('ConvLSTM_metrics.npy')
logGauss_metrics = np.load('logGauss_metrics.npy')
Chawkes_metrics = np.load('Chawkes_metrics.npy')
Proposed_metrics = np.load('Proposed_metrics.npy')
ST_Hawkes_metrics = np.load('ST_Hawkes_metrics.npy')
multimodalST_Hawkes_metrics = np.load('multimodalST_Hawkes_metrics.npy')

Np0 = 100
all_MAEmean = np.zeros((7,))
all_MAEmean[0] = Thawkes_metrics[0,:Np0].mean()
all_MAEmean[1] = ConvLSTM_metrics[0,:Np0].mean()
all_MAEmean[2] = logGauss_metrics[0,:Np0].mean()
all_MAEmean[3] = Chawkes_metrics[0,:Np0].mean()
all_MAEmean[4] = Proposed_metrics[0,:Np0].mean()
all_MAEmean[5] = ST_Hawkes_metrics[0,:Np0].mean()
all_MAEmean[6] = multimodalST_Hawkes_metrics[0,:Np0].mean()
print('MAE mean = ',all_MAEmean)


all_MAEstd = np.zeros((7,))
all_MAEstd[0] = Thawkes_metrics[0,:Np0].std()
all_MAEstd[1] = ConvLSTM_metrics[0,:Np0].std()
all_MAEstd[2] = logGauss_metrics[0,:Np0].std()
all_MAEstd[3] = Chawkes_metrics[0,:Np0].std()
all_MAEstd[4] = Proposed_metrics[0,:Np0].std()
all_MAEstd[5] = ST_Hawkes_metrics[0,:Np0].std()
all_MAEstd[6] = multimodalST_Hawkes_metrics[0,:Np0].std()
print('MAE std = ',all_MAEstd)


all_RMAEmean = np.zeros((7,))
all_RMAEmean[0] = Thawkes_metrics[1,:Np0].mean()
all_RMAEmean[1] = ConvLSTM_metrics[1,:Np0].mean()
all_RMAEmean[2] = logGauss_metrics[1,:Np0].mean()
all_RMAEmean[3] = Chawkes_metrics[1,:Np0].mean()
all_RMAEmean[4] = Proposed_metrics[1,:Np0].mean()
all_RMAEmean[5] = ST_Hawkes_metrics[1,:Np0].mean()
all_RMAEmean[6] = multimodalST_Hawkes_metrics[1,:Np0].mean()
print('RMAE mean = ',all_RMAEmean)


all_RMAEstd = np.zeros((7,))
all_RMAEstd[0] = Thawkes_metrics[1,:Np0].std()
all_RMAEstd[1] = ConvLSTM_metrics[1,:Np0].std()
all_RMAEstd[2] = logGauss_metrics[1,:Np0].std()
all_RMAEstd[3] = Chawkes_metrics[1,:Np0].std()
all_RMAEstd[4] = Proposed_metrics[1,:Np0].std()
all_RMAEstd[5] = ST_Hawkes_metrics[1,:Np0].std()
all_RMAEstd[6] = multimodalST_Hawkes_metrics[1,:Np0].std()
print('RMAE std = ',all_RMAEstd)


all_MAE_CIFmean = np.zeros((7,))
all_MAE_CIFmean[0] = Thawkes_metrics[3,:Np0].mean()
all_MAE_CIFmean[1] = ConvLSTM_metrics[3,:Np0].mean()
all_MAE_CIFmean[2] = logGauss_metrics[3,:Np0].mean()
all_MAE_CIFmean[3] = Chawkes_metrics[3,:Np0].mean()
all_MAE_CIFmean[4] = Proposed_metrics[3,:Np0].mean()
all_MAE_CIFmean[5] = ST_Hawkes_metrics[3,:Np0].mean()
all_MAE_CIFmean[6] = multimodalST_Hawkes_metrics[3,:Np0].mean()
print('MAE mean = ',all_MAE_CIFmean)


all_MAE_CIFstd = np.zeros((7,))
all_MAE_CIFstd[0] = Thawkes_metrics[3,:Np0].std()
all_MAE_CIFstd[1] = ConvLSTM_metrics[3,:Np0].std()
all_MAE_CIFstd[2] = logGauss_metrics[3,:Np0].std()
all_MAE_CIFstd[3] = Chawkes_metrics[3,:Np0].std()
all_MAE_CIFstd[4] = Proposed_metrics[3,:Np0].std()
all_MAE_CIFstd[5] = ST_Hawkes_metrics[3,:Np0].std()
all_MAE_CIFstd[6] = multimodalST_Hawkes_metrics[3,:Np0].std()
print('MAE std = ',all_MAE_CIFstd)


all_RMAE_CIFmean = np.zeros((7,))
all_RMAE_CIFmean[0] = Thawkes_metrics[4,:Np0].mean()
all_RMAE_CIFmean[1] = ConvLSTM_metrics[4,:Np0].mean()
all_RMAE_CIFmean[2] = logGauss_metrics[4,:Np0].mean()
all_RMAE_CIFmean[3] = Chawkes_metrics[4,:Np0].mean()
all_RMAE_CIFmean[4] = Proposed_metrics[4,:Np0].mean()
all_RMAE_CIFmean[5] = ST_Hawkes_metrics[4,:Np0].mean()
all_RMAE_CIFmean[6] = multimodalST_Hawkes_metrics[4,:Np0].mean()
print('MAE mean = ',all_RMAE_CIFmean)


all_RMAE_CIFstd = np.zeros((7,))
all_RMAE_CIFstd[0] = Thawkes_metrics[4,:Np0].std()
all_RMAE_CIFstd[1] = ConvLSTM_metrics[4,:Np0].std()
all_RMAE_CIFstd[2] = logGauss_metrics[4,:Np0].std()
all_RMAE_CIFstd[3] = Chawkes_metrics[4,:Np0].std()
all_RMAE_CIFstd[4] = Proposed_metrics[4,:Np0].std()
all_RMAE_CIFstd[5] = ST_Hawkes_metrics[4,:Np0].std()
all_RMAE_CIFstd[6] = multimodalST_Hawkes_metrics[4,:Np0].std()
print('MAE std = ',all_RMAE_CIFstd)


all_Timemean = np.zeros((7,))
all_Timemean[0] = Thawkes_metrics[2,:Np0].mean()
all_Timemean[1] = ConvLSTM_metrics[2,:Np0].mean()
all_Timemean[2] = logGauss_metrics[2,:Np0].mean()
all_Timemean[3] = Chawkes_metrics[2,:Np0].mean()
all_Timemean[4] = Proposed_metrics[2,:Np0].mean()
all_Timemean[5] = ST_Hawkes_metrics[2,:Np0].mean()
all_Timemean[6] = multimodalST_Hawkes_metrics[2,:Np0].mean()
print('MAE mean = ',all_Timemean)


all_Timestd = np.zeros((7,))
all_Timestd[0] = Thawkes_metrics[2,:Np0].std()
all_Timestd[1] = ConvLSTM_metrics[2,:Np0].std()
all_Timestd[2] = logGauss_metrics[2,:Np0].std()
all_Timestd[3] = Chawkes_metrics[2,:Np0].std()
all_Timestd[4] = Proposed_metrics[2,:Np0].std()
all_Timestd[5] = ST_Hawkes_metrics[2,:Np0].std()
all_Timestd[6] = multimodalST_Hawkes_metrics[2,:Np0].std()
print('MAE std = ',all_Timestd)





