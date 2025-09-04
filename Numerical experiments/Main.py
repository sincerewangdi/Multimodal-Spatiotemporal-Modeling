#proposed method

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


Proposed_Theta = np.load('Proposed_Theta.npy')
Proposed_Phi = np.load('Proposed_Phi.npy')
Proposed_Tal = np.load('Proposed_Tal.npy')
Proposed_Z = np.load('Proposed_Z.npy')
Proposed_Mu = np.load('Proposed_Mu.npy')
Proposed_Alpha = np.load('Proposed_Alpha.npy')
Proposed_Gamma = np.load('Proposed_Gamma.npy')
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_train_CIF = np.load('Proposed_flow_train_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
Proposed_hat_flow = np.load('Proposed_hat_flow.npy')
Proposed_metrics = np.load('Proposed_metrics.npy')



for ip in range(0,Np):
    ip
    #Generate data
    st_info, flow_train, flow_train_CIF, flow_test, flow_test_CIF = data_generating()
    #Model
    All_Theta,All_Phi,All_Tal,All_Z,All_Mu,All_Alpha,All_Gamma,Time = model_proposed(flow_train,st_info)
    #
    #model testing
    hat_flow, MAE,RMAE,MAE_CIF,RMAE_CIF = model_proposed_test(flow_test,flow_test_CIF,All_Mu,All_Alpha)
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



    
#Save results
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



##########################################################
#Figures


#############################
#Original data
Proposed_flow_train = np.load('Proposed_flow_train.npy')
Proposed_flow_train_CIF = np.load('Proposed_flow_train_CIF.npy')
Proposed_st_info = np.load('Proposed_st_info.npy')

#Figure 4
#Fig 4 a): Event frequency data
ip = 0
flow_train = Proposed_flow_train[:,:,:,:,ip]
flow_train_CIF = Proposed_flow_train_CIF[:,:,:,:,ip]
st_info = Proposed_st_info[:,:,ip]

day = 0
vmin0 = flow_train[:,:,day,:].min()
vmax0 = flow_train[:,:,day,:].max()
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
pyplot.subplot(3,1,1)
sns.heatmap(flow_train[:,:,day,0].T,linewidths=.01,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
pyplot.title('Layer 1')
x_labels = np.array([i for i in range(1,flow_train[:,:,day,0].shape[0]+1,5)])
pyplot.xticks(np.arange(1,flow_train[:,:,day,0].shape[0]+1,5), x_labels)
y_labels = np.array([i for i in range(1,flow_train[:,:,day,0].shape[1]+1,3)])
pyplot.yticks(np.arange(1,flow_train[:,:,day,0].shape[1]+1,3), y_labels)


pyplot.subplot(3,1,2)
sns.heatmap(flow_train[:,:,day,1].T,linewidths=.01,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
pyplot.title('Layer 2')
x_labels = np.array([i for i in range(1,flow_train[:,:,day,1].shape[0]+1,5)])
pyplot.xticks(np.arange(1,flow_train[:,:,day,1].shape[0]+1,5), x_labels)
y_labels = np.array([i for i in range(1,flow_train[:,:,day,1].shape[1]+1,3)])
pyplot.yticks(np.arange(1,flow_train[:,:,day,1].shape[1]+1,3), y_labels)


pyplot.subplot(3,1,3)
sns.heatmap(flow_train[:,:,day,2].T,linewidths=.01,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
pyplot.title('Layer 3')
x_labels = np.array([i for i in range(1,flow_train[:,:,day,1].shape[0]+1,5)])
pyplot.xticks(np.arange(1,flow_train[:,:,day,1].shape[0]+1,5), x_labels)
y_labels = np.array([i for i in range(1,flow_train[:,:,day,1].shape[1]+1,3)])
pyplot.yticks(np.arange(1,flow_train[:,:,day,1].shape[1]+1,3), y_labels)

pyplot.show()



#Fig 4 b): attribute data
sns.heatmap(st_info.T,linewidths=.1,cmap="binary_r")
pyplot.ylabel('Attribute')
pyplot.xlabel('Node')
x_labels = np.array([i for i in range(1,st_info.shape[0]+1,3)])
pyplot.xticks(np.arange(1,st_info.shape[0]+1,3), x_labels)
y_labels = np.array([i for i in range(1,st_info.shape[1]+1,3)])
pyplot.yticks(np.arange(1,st_info.shape[1]+1,3), y_labels)
pyplot.show()


#Fig 4 c): ACF of the time sequence at node 1 and layer 1
import statsmodels.api as sm
l = 0
st = 0
data = flow_train[:,st,day,l]
sm.graphics.tsa.plot_acf(data,lags = 20)
pyplot.xlabel('Lag')
pyplot.ylabel('ACF')
pyplot.show()

#Fig 4 d): Pearson correlation coefficient among nodes at layer 1

pearson_space = np.zeros((Num_st,Num_st,L))
for l in range(0,L):
    for st_i in range(0,Num_st):
        for st_j in range(0,Num_st):
            data_s1 = np.reshape(flow_train[:,st_i,:,l],((t_T-t_1) * Num_month_train,))
            data_s2 = np.reshape(flow_train[:,st_j,:,l],((t_T-t_1) * Num_month_train,))             
            pr1,pr2 = scipy.stats.pearsonr(data_s1,data_s2)
            pearson_space[st_i, st_j,l] = pr1


#layer 1: spatial correlation
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(pearson_space[:,:,0],linewidths=.01,cmap="YlGnBu")
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,pearson_space[:,:,0].shape[1]+1,3)])
pyplot.xticks(np.arange(1,pearson_space[:,:,0].shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,pearson_space[:,:,0].shape[0]+1,3)])
pyplot.yticks(np.arange(1,pearson_space[:,:,0].shape[0]+1,3), y_labels)
pyplot.show()


#Fig 4 e): Pearson correlation coefficient among layers
pearson_layer = np.zeros((L,L))
for l1 in range(0,L):
    for l2 in range(0,L):
        data_l1 = np.reshape(flow_train[:,:,:,l1],(Num_st * (t_T-t_1) * Num_month_train,))
        data_l2 = np.reshape(flow_train[:,:,:,l2],(Num_st * (t_T-t_1) * Num_month_train,))              
        pr1,pr2 = scipy.stats.pearsonr(data_l1,data_l2)
        pearson_layer[l1,l2] = pr1


pyplot.figure(figsize=(4.5, 4))
sns.heatmap(pearson_layer,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Layer')
pyplot.ylabel('Layer')
x_labels = np.array([i for i in range(1,pearson_layer.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,pearson_layer.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()


#############################
#Parameters
rTal = np.load('rTal.npy')
rTheta = np.load('rTheta.npy')
rGamma = np.load('rGamma.npy')
rPhi = np.load('rPhi.npy')
rMu = np.load('rMu.npy')
rAlpha = np.load('rAlpha.npy')


Proposed_Theta = np.load('Proposed_Theta.npy')
Proposed_Phi = np.load('Proposed_Phi.npy')
Proposed_Tal = np.load('Proposed_Tal.npy')
Proposed_Z = np.load('Proposed_Z.npy')
Proposed_Mu = np.load('Proposed_Mu.npy')
Proposed_Alpha = np.load('Proposed_Alpha.npy')
Proposed_Gamma = np.load('Proposed_Gamma.npy')


Np = 100
P_Theta = np.ones((R,R,Np))
P_Phi = np.ones((R,K,Np))
P_Tal = np.zeros((Num_st,R,Np))
P_Z = np.zeros((Num_st,R,Np))
P_Mu = np.zeros((Num_st,t_T-t_1,L,Np))
P_Alpha = np.zeros((Num_st,Num_st,L,Np))
P_Gamma = np.zeros((Num_st,Num_st,Np))

for ip in range(0,Np):
    All_Mu = Proposed_Mu[:,:,:,:,ip]
    Nend = (np.argwhere((All_Mu.sum(axis=0).sum(axis=0).sum(axis=0)==0))[:,0]).min()-1
    Mu0 = All_Mu[:,:,:,Nend]
    Theta0 = Proposed_Theta[:,:,Nend,ip]
    Phi0 = Proposed_Phi[:,:,Nend,ip]
    Tal0 = Proposed_Tal[:,:,Nend,ip]
    Z0 = Proposed_Z[:,:,Nend,ip]
    Alpha0 = Proposed_Alpha[:,:,:,Nend,ip]
    Gamma0 = Proposed_Gamma[:,:,Nend,ip]
    P_Mu[:,:,:,ip] = Mu0
    P_Theta[:,:,ip] = Theta0
    P_Phi[:,:,ip] = Phi0
    P_Tal[:,:,ip] = Tal0
    P_Z[:,:,ip] = Z0
    P_Alpha[:,:,:,ip] = Alpha0
    P_Gamma[:,:,ip] = Gamma0



#Spatial information
Mu = P_Mu.mean(axis = 3)
Theta = P_Theta.mean(axis = 2)
Phi = P_Phi.mean(axis = 2)
Tal = P_Tal.mean(axis = 2)
Z = P_Z.mean(axis = 2)
Alpha = P_Alpha.mean(axis = 3)
Gamma = P_Gamma.mean(axis = 2)




#Figure 6 a) and b): Theta
rTheta = np.zeros((L,L))
rTheta[0,:] = [1, 0.21428571, 0.08333333]
rTheta[1,:] = [0.21428571, 1, 0.31746032]
rTheta[2,:] = [0.08333333, 0.31746032, 1]

vmax0 = max(Theta.max(),rTheta.max())
vmin0 = min(Theta.min(),rTheta.min())

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Theta,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Community')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Theta.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Theta.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(rTheta,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Community')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Theta.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Theta.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()





#Figure 6 c) and d): Phi
vmax0 = max(Phi.max(),rPhi.max())
vmin0 = min(Phi.min(),rPhi.min())

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Phi,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Attribute')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Phi.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Phi.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(rPhi,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Attribute')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Phi.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Phi.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()




#Figure 6 e) and f): Tal
vmax0 = max(Tal.max(),rTal.max())
vmin0 = min(Tal.min(),rTal.min())

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Tal,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Community')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,Tal.shape[0]+1,3)])
pyplot.yticks(np.arange(1,Tal.shape[0]+1,3), y_labels)
x_labels = np.array([i for i in range(1,Tal.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
pyplot.show()

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(rTal,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Community')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,Tal.shape[0]+1,3)])
pyplot.yticks(np.arange(1,Tal.shape[0]+1,3), y_labels)
x_labels = np.array([i for i in range(1,Tal.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
pyplot.show()





#Figure 7: Mu
l = 2 #0,1,2
vmax0 = max(Mu[:,:,l].max(),rMu[:,:,l].max())
vmin0 = min(Mu[:,:,l].min(),rMu[:,:,l].min())

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Mu[:,:,l],linewidths=0,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Mu[:,:,l].shape[1]+1,5)])
pyplot.xticks(np.arange(1,Mu[:,:,l].shape[1]+1,5), x_labels)
y_labels = np.array([i for i in range(1,Mu[:,:,l].shape[0]+1,3)])
pyplot.yticks(np.arange(1,Mu[:,:,l].shape[0]+1,3), y_labels)
pyplot.show()





#Figure 8: Gamma
vmax0 = max(Gamma.max(),rGamma.max())
vmin0 = min(Gamma.min(),rGamma.min())

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Gamma,linewidths=.1,cmap="YlGnBu",norm = norm)  #cmap="YlGnBu"
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Gamma.shape[1]+1,3)])
pyplot.xticks(np.arange(1,Gamma.shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,Gamma.shape[0]+1,3)])
pyplot.yticks(np.arange(1,Gamma.shape[0]+1,3), y_labels)
pyplot.show()

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(rGamma,linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,rGamma.shape[1]+1,3)])
pyplot.xticks(np.arange(1,rGamma.shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,rGamma.shape[0]+1,3)])
pyplot.yticks(np.arange(1,rGamma.shape[0]+1,3), y_labels)
pyplot.show()



#Figure 9: Alpha
l = 0 #0,1,2
vmax0 = max(Alpha[:,:,l].max(),rAlpha[:,:,l].max())
vmin0 = min(Alpha[:,:,l].min(),rAlpha[:,:,l].min())

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Alpha[:,:,l],linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Alpha[:,:,l].shape[1]+1,3)])
pyplot.xticks(np.arange(1,Alpha[:,:,l].shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,Alpha[:,:,l].shape[0]+1,3)])
pyplot.yticks(np.arange(1,Alpha[:,:,l].shape[0]+1,3), y_labels)
pyplot.show()

pyplot.figure(figsize=(4.5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(rAlpha[:,:,l],linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,rAlpha[:,:,l].shape[1]+1,3)])
pyplot.xticks(np.arange(1,rAlpha[:,:,l].shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,rAlpha[:,:,l].shape[0]+1,3)])
pyplot.yticks(np.arange(1,rAlpha[:,:,l].shape[0]+1,3), y_labels)
pyplot.show()




#Metrics
#Table 1: RelErr
RelErr_Gamma = abs(rGamma - Gamma).mean() / rGamma.mean()
RelErr_Mu = abs(rMu - Mu).mean() / rMu.mean()
RelErr_Alpha = abs(rAlpha - Alpha).mean() / rAlpha.mean()
RelErr_Phi = abs(rPhi - Phi).mean() / rPhi.mean()
RelErr_Theta = abs(rTheta - Theta).mean() / rTheta.mean()
RelErr_Tal = abs(rTal - Tal).mean() / rTal.mean()


print('RelErr_Gamma=',RelErr_Gamma) #0.027647058823529413
print('RelErr_Mu=',RelErr_Mu) #0.0052392817804207
print('RelErr_Alpha=',RelErr_Alpha) #0.043429379253688075
print('RelErr_Phi=',RelErr_Phi) #0.007457727072310447
print('RelErr_Theta=',RelErr_Theta) #0.03615384541515375
print('RelErr_Tal=',RelErr_Tal) #0.0

    

#############################
#Figure: convergence
Proposed_Gamma = np.load('Proposed_Gamma.npy')
conv = Proposed_Gamma.mean(axis = 0).mean(axis=0)
pyplot.figure(figsize=(5, 4))
for ic in range(0,conv.shape[1]):
    nc = np.argwhere(conv[:,ic] > 0)[:,0]
    pyplot.plot(nc, conv[nc,ic], 'r+-')



pyplot.xlabel('Iteration')
pyplot.ylabel('$\Gamma$')
pyplot.grid(linestyle = ':')
pyplot.show()





#Figure B1: Examples of predicted event frequencies by the proposed method and corresponding true count and CIF values

ip = 0
Proposed_flow_test = np.load('Proposed_flow_test.npy')
Proposed_hat_flow = np.load('Proposed_hat_flow.npy')
Proposed_flow_test_CIF = np.load('Proposed_flow_test_CIF.npy')
flow_test = Proposed_flow_test[:,:,:,:,ip]
flow_test_CIF = Proposed_flow_test_CIF[:,:,:,:,ip]
hat_flow = Proposed_hat_flow[:,:,:,:,ip]



st = 2 #2,15,16
l = 0 #0,1,2
day = 10
time = np.array([i for i in range(1,flow_test[:,st,day,l].shape[0]+1)])
pyplot.figure(figsize=(4.5, 4))
pyplot.plot(time,flow_test[:,st,day,l],'.',color='red',linewidth = '1',label = 'Count')
pyplot.plot(time,flow_test_CIF[:,st,day,l],color='blue',linewidth = '1',label = 'CIF')
pyplot.plot(time,hat_flow[:,st,day,l],color='green',linewidth = '2',label = 'Proposed')
pyplot.legend()
pyplot.xlabel('Time')
pyplot.grid(linestyle = ':')
pyplot.show()




