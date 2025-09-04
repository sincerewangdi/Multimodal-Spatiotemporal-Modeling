#Figures

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
#Data
Demand0 = np.load('Demand.npy')
st_info0 = np.load('Attr.npy')
num0 = [0,3,4,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21]
Demand = Demand0[:,num0,:,:]
st_info = st_info0[num0,:]

Num_month_pre_weekday = Demand.shape[2]

#Figure D1 a) and b)
st = 6 #6,15
time = np.array([i for i in range(1,Demand[:,st,iday,0].shape[0]+1)])
pyplot.subplot(2,1,1)
for iday in range(0,Num_month_pre_weekday):
    pyplot.plot(Demand[:,st,iday,0])
    pyplot.ylabel('Origin demand')
    pyplot.xlabel('Time')


pyplot.grid(linestyle = ':')
pyplot.subplot(2,1,2)
for iday in range(0,Num_month_pre_weekday):
    pyplot.plot(Demand[:,st,iday,1])
    pyplot.ylabel('Destination demand')
    pyplot.xlabel('Time')


pyplot.grid(linestyle = ':')
pyplot.show()


#Figure D1 c)
sns.heatmap(st_info.T,cmap="YlGnBu")
pyplot.ylabel('Attribute')
pyplot.xlabel('Node')
x_labels = np.array([i for i in range(1,st_info.shape[0]+1,3)])
pyplot.xticks(np.arange(1,st_info.shape[0]+1,3), x_labels)
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
numa = int(np.load('numa.npy'))
Mu = All_Mu[:,:,:,numa] 
Alpha = All_Alpha[:,:,:,numa] 
Gamma = All_Gamma[:,:,numa]
Theta = All_Theta[:,:,numa]
Phi = All_Phi[:,:,numa]
Tal = All_Tal[:,:,numa]




#Figure D2 a): Tal
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(Tal,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Community')
pyplot.ylabel('Node')
y_labels = np.array([i for i in range(1,Tal.shape[0]+1,3)])
pyplot.yticks(np.arange(1,Tal.shape[0]+1,3), y_labels)
x_labels = np.array([i for i in range(1,Tal.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
pyplot.show()

#Figure D2 b): Phi
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(Phi,linewidths=.1,cmap="YlGnBu",annot=True,)
pyplot.xlabel('Attribute')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Phi.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Phi.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()


#Figure D2 c): Theta
pyplot.figure(figsize=(4.5, 4))
sns.heatmap(Theta,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Community')
pyplot.ylabel('Community')
x_labels = np.array([i for i in range(1,Theta.shape[1]+1)])
pyplot.xticks(range(len(x_labels)),x_labels)
y_labels = np.array([i for i in range(1,Theta.shape[0]+1)])
pyplot.yticks(range(len(y_labels)),y_labels)
pyplot.show()



#Table D3 a) and b): Mu
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


#Table D3 c) and d): time-varying Mu
i = 6 #6, 15
time = np.array([i for i in range(1,Mu[i,:,0].shape[0]+1)])
pyplot.figure(figsize=(4.5, 4))
pyplot.plot(time,Mu[i,:,0],color='red',linewidth = '1',label = 'Layer 1')
pyplot.plot(time,Mu[i,:,1],'--',color='blue',linewidth = '1',label = 'Layer 2')
pyplot.legend()
pyplot.xlabel('Time')
pyplot.ylabel('$\mu$')
pyplot.grid(linestyle = ':')
pyplot.show()


#Table D4 a): Gamma
pyplot.figure(figsize=(5, 4))
sns.heatmap(Gamma,linewidths=.1,cmap="YlGnBu")
pyplot.xlabel('Node')
pyplot.ylabel('Node')
x_labels = np.array([i for i in range(1,Gamma.shape[1]+1,5)])
pyplot.xticks(np.arange(1,Gamma.shape[1]+1,5), x_labels)
y_labels = np.array([i for i in range(1,Gamma.shape[0]+1,5)])
pyplot.yticks(np.arange(1,Gamma.shape[0]+1,5), y_labels)
pyplot.show()


#Table D4 b): Alpha
vmax0 = 0.5 #Alpha.max()
vmin0 = Alpha.min()
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

#Table D4 c): 
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

