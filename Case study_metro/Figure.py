# Figures
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



##############################################################
#Data presentation
Inflow_pre_weekday = np.load('Inflow_pre_weekday.npy') #(234, 91, 40)
Outflow_pre_weekday = np.load('Outflow_pre_weekday.npy') #(234, 91, 40)
stationID = np.load('stationID.npy')
Num_month_pre_weekday = Inflow_pre_weekday.shape[2]

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


#Figure 11 a) and b)
st = 30 #29,59 station number
time = np.array([i for i in range(1,Inflow_pre_weekday.shape[0]+1)])
pyplot.subplot(2,1,1)
for iday in range(0,Num_month_pre_weekday):
    pyplot.plot(time,Inflow_pre_weekday[:,st,iday])
    pyplot.ylabel('Inflow')
    pyplot.xlabel('Time')


pyplot.grid(linestyle = ':')
pyplot.subplot(2,1,2)
for iday in range(0,Num_month_pre_weekday):
    pyplot.plot(time,Outflow_pre_weekday[:,st,iday])
    pyplot.ylabel('Outflow')
    pyplot.xlabel('Time')


pyplot.grid(linestyle = ':')
pyplot.show()


##Figure 11 c)
st_info = np.load('st_info_pre.npy')
Num_time_pre = flow_train.shape[0]
Num_st = flow_train.shape[1]
t_T = Num_time_pre
t_1 = 0
K = st_info.shape[1] #number of attributes
L = 2 #layers
delta = 1
ax = sns.heatmap(st_info.T,cmap="YlGnBu")
pyplot.ylabel('Attribute')
pyplot.xlabel('Node')
x_labels = np.array([i for i in range(1,st_info.shape[0]+1,5)])
pyplot.xticks(np.arange(1,st_info.shape[0]+1,5), x_labels)
y_labels = np.array([i for i in range(1,st_info.shape[1]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()





##############################################################
#Parameter estimation
All_Theta = np.load('All_Theta.npy')
All_Phi = np.load('All_Phi.npy')
All_Tal = np.load('All_Tal.npy')
All_Mu = np.load('All_Mu.npy')
All_Alpha = np.load('All_Alpha.npy')
All_Gamma = np.load('All_Gamma.npy')
numa = 38
Mu = All_Mu[:,:,:,numa] 
Alpha = All_Alpha[:,:,:,numa] 
Gamma = All_Gamma[:,:,numa]
Theta = All_Theta[:,:,numa]
Phi = All_Phi[:,:,numa]
Tal = All_Tal[:,:,numa]

def fun_u(x):
    if x>0:
        y = 1
    else:
        y = 0
    return y



#Table C2
R = Theta.shape[0]
ETA = np.zeros((R,R))
for r1 in range(0,R):
    for r2 in range(0,R):
        ETA[r1,r2] = fun_u(Theta[r1,r1]-Theta[r1,r2]) + fun_u(Theta[r1,r1]-Theta[r2,r1])


eta = sum(sum(ETA))/(2*R*(R-1))




#Figure 12 a): Tal
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(Tal,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Community')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,Tal.shape[0]+1,5)])
pyplot.yticks(np.arange(1,Tal.shape[0]+1,5), y_labels)
x_labels = np.array([i for i in range(1,Tal.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
pyplot.show()


#Figure 12 b): Phi
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(Phi,linewidths=.1,cmap="YlGnBu",annot=True,)
pyplot.xlabel('Attribute')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Phi.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Phi.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()


#Figure 12 c): Theta
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(Theta,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Community')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Theta.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Theta.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()





#Figure 13 a) and b): Mu
vmax0 = Mu.max()
vmin0 = Mu.min()
pyplot.figure(figsize=(6, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Mu[:,:,0],linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Mu[:,:,0].shape[1]+1,3)])
pyplot.xticks(np.arange(1,Mu[:,:,0].shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,Mu[:,:,0].shape[0]+1,5)])
pyplot.yticks(np.arange(1,Mu[:,:,0].shape[0]+1,5), y_labels)
pyplot.show()

pyplot.figure(figsize=(6, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Mu[:,:,1],linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Mu[:,:,1].shape[1]+1,3)])
pyplot.xticks(np.arange(1,Mu[:,:,1].shape[1]+1,3), x_labels)
y_labels = np.array([i for i in range(1,Mu[:,:,1].shape[0]+1,5)])
pyplot.yticks(np.arange(1,Mu[:,:,1].shape[0]+1,5), y_labels)
pyplot.show()



#Figure 13 c) - f): time-varying Mu
i = 0 # 0,20,40,70
time = np.array([i for i in range(1,Mu[i,:,0].shape[0]+1)])
pyplot.figure(figsize=(4.5, 4))
pyplot.plot(time,Mu[i,:,0],color='red',linewidth = '1',label = 'Layer 1')
pyplot.plot(time,Mu[i,:,1],'--',color='blue',linewidth = '1',label = 'Layer 2')
pyplot.legend()
pyplot.xlabel('Time')
pyplot.ylabel('$\mu$')
pyplot.grid(linestyle = ':')
pyplot.show()




#Figure 14 a): Gamma
pyplot.figure(figsize=(5, 4))
sns.heatmap(Gamma,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Gamma.shape[1]+1,5)])
pyplot.xticks(np.arange(1,Gamma.shape[1]+1,5), x_labels)
y_labels = np.array([i for i in range(1,Gamma.shape[0]+1,5)])
pyplot.yticks(np.arange(1,Gamma.shape[0]+1,5), y_labels)
pyplot.show()



#Figure 14 b) and c): Alpha
vmax0 = 0.02#Alpha.max()
vmin0 = 0#Alpha.min()
pyplot.figure(figsize=(5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Alpha[:,:,0],linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Alpha[:,:,0].shape[1]+1,5)])
pyplot.xticks(np.arange(1,Alpha[:,:,0].shape[1]+1,5), x_labels)
y_labels = np.array([i for i in range(1,Alpha[:,:,0].shape[0]+1,5)])
pyplot.yticks(np.arange(1,Alpha[:,:,0].shape[0]+1,5), y_labels)
pyplot.show()

pyplot.figure(figsize=(5, 4))
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
sns.heatmap(Alpha[:,:,1],linewidths=.1,cmap="YlGnBu",norm = norm)
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Alpha[:,:,1].shape[1]+1,5)])
pyplot.xticks(np.arange(1,Alpha[:,:,1].shape[1]+1,5), x_labels)
y_labels = np.array([i for i in range(1,Alpha[:,:,1].shape[0]+1,5)])
pyplot.yticks(np.arange(1,Alpha[:,:,1].shape[0]+1,5), y_labels)
pyplot.show()



#################################################
hat_flow1 = np.load('Proposed_hat_flow1.npy')
hat_flow2 = np.load('Proposed_hat_flow2.npy')
hat_flow3 = np.load('Proposed_hat_flow3.npy')



#Figure 15
st = 86 #12 25 45 86
day = 10
l = 0
time = np.array([i for i in range(1,Num_time_pre+1)])
pyplot.figure(figsize=(6, 3))
pyplot.plot(time, flow_test[:,st,day,l],'.',color='red',linewidth = '1',label = 'Inflow data')
pyplot.plot(time[1:],hat_flow1[1:,st,day,l],'-',color='green',linewidth = '1',label = 'One-step forecast')
pyplot.plot(time[2:],hat_flow2[2:,st,day,l],'-.',color='blue',linewidth = '1',label = 'Two-step forecast')
pyplot.plot(time[3:],hat_flow3[3:,st,day,l],'--',color='black',linewidth = '1',label = 'Three-step forecast')
pyplot.legend()
pyplot.xlabel('Time')
pyplot.ylabel('Flow')
pyplot.grid(linestyle = ':')
pyplot.show()


l = 1
pyplot.figure(figsize=(6, 3))
pyplot.plot(time, flow_test[:,st,day,l],'.',color='red',linewidth = '1',label = 'Outflow data')
pyplot.plot(time[1:],hat_flow1[1:,st,day,l],'-',color='green',linewidth = '1',label = 'One-step forecast')
pyplot.plot(time[2:],hat_flow2[2:,st,day,l],'-.',color='blue',linewidth = '1',label = 'Two-step forecast')
pyplot.plot(time[3:],hat_flow3[3:,st,day,l],'--',color='black',linewidth = '1',label = 'Three-step forecast')
pyplot.legend()
pyplot.xlabel('Time')
pyplot.ylabel('Flow')
pyplot.grid(linestyle = ':')
pyplot.show()


#Figure C2
day = 10
vmin0 = flow_test[:,:,day,:].min()
vmax0 = flow_test[:,:,day,:].max()
norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
pyplot.subplot(2,1,1)
sns.heatmap(flow_test[:,:,day,0].T,linewidths=.01,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,flow_test[:,:,day,0].shape[1]+1,10)])
pyplot.yticks(np.arange(1,flow_test[:,:,day,0].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,flow_test[:,:,day,0].shape[0]+1,3)])
pyplot.xticks(np.arange(1,flow_test[:,:,day,0].shape[0]+1,3), x_labels)
pyplot.title('Layer 1')
pyplot.subplot(2,1,2)
sns.heatmap(flow_test[:,:,day,1].T,linewidths=.01,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,flow_test[:,:,day,1].shape[1]+1,10)])
pyplot.yticks(np.arange(1,flow_test[:,:,day,1].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,flow_test[:,:,day,1].shape[0]+1,3)])
pyplot.xticks(np.arange(1,flow_test[:,:,day,1].shape[0]+1,3), x_labels)
pyplot.title('Layer 2')
pyplot.show()


diff1 = hat_flow1 - flow_test
diff2 = hat_flow2 - flow_test
diff3 = hat_flow3 - flow_test
vmin01 = diff1[:,:,day,:].min()
vmax01 = diff1[:,:,day,:].max()
vmin02 = diff2[:,:,day,:].min()
vmax02 = diff2[:,:,day,:].max()
vmin03 = diff3[:,:,day,:].min()
vmax03 = diff3[:,:,day,:].max()
vmin0 = min(min(vmin01,vmin02),vmin03)
vmax0 = max(max(vmax01,vmax02),vmax03)

norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
pyplot.subplot(2,1,1)
sns.heatmap(diff1[:,:,day,0].T,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,diff1[:,:,day,0].shape[1]+1,10)])
ymajorLocator = MultipleLocator(10)
ax.yaxis.set_major_locator(ymajorLocator)
pyplot.yticks(np.arange(1,diff1[:,:,day,0].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,diff1[:,:,day,0].shape[0]+1,3)])
pyplot.xticks(np.arange(1,diff1[:,:,day,0].shape[0]+1,3), x_labels)
pyplot.title('Layer 1')
pyplot.subplot(2,1,2)
sns.heatmap(diff1[:,:,day,1].T,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,diff1[:,:,day,1].shape[1]+1,10)])
ymajorLocator = MultipleLocator(10)
x_labels = np.array([i for i in range(1,diff1[:,:,day,1].shape[0]+1,3)])
xmajorLocator = MultipleLocator(3)
ax.xaxis.set_major_locator(xmajorLocator)
pyplot.xticks(np.arange(1,diff1[:,:,day,0].shape[0]+1,3), x_labels)
pyplot.title('Layer 2')
pyplot.show()




norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
pyplot.subplot(2,1,1)
sns.heatmap(diff2[:,:,day,0].T,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,diff2[:,:,day,0].shape[1]+1,10)])
pyplot.yticks(np.arange(1,diff2[:,:,day,0].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,diff2[:,:,day,0].shape[0]+1,3)])
pyplot.xticks(np.arange(1,diff2[:,:,day,0].shape[0]+1,3), x_labels)
pyplot.title('Layer 1')
pyplot.subplot(2,1,2)
sns.heatmap(diff2[:,:,day,1].T,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,diff2[:,:,day,1].shape[1]+1,10)])
pyplot.yticks(np.arange(1,diff2[:,:,day,1].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,diff2[:,:,day,1].shape[0]+1,3)])
pyplot.xticks(np.arange(1,diff2[:,:,day,0].shape[0]+1,3), x_labels)
pyplot.title('Layer 2')
pyplot.show()




norm = matplotlib.colors.Normalize(vmin=vmin0,vmax=vmax0)
pyplot.subplot(2,1,1)
sns.heatmap(diff3[:,:,day,0].T,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,diff3[:,:,day,0].shape[1]+1,10)])
pyplot.yticks(np.arange(1,diff3[:,:,day,0].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,diff3[:,:,day,0].shape[0]+1,3)])
pyplot.xticks(np.arange(1,diff3[:,:,day,0].shape[0]+1,3), x_labels)
pyplot.title('Layer 1')
pyplot.subplot(2,1,2)
sns.heatmap(diff3[:,:,day,1].T,cmap="YlGnBu",norm=norm)
pyplot.xlabel('Time')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,diff3[:,:,day,1].shape[1]+1,10)])
pyplot.yticks(np.arange(1,diff3[:,:,day,1].shape[1]+1,10), y_labels)
x_labels = np.array([i for i in range(1,diff3[:,:,day,1].shape[0]+1,3)])
pyplot.xticks(np.arange(1,diff3[:,:,day,1].shape[0]+1,3), x_labels)
pyplot.title('Layer 2')
pyplot.show()

