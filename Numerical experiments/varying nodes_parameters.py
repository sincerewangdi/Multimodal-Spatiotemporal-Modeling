# Table B2, Varying nodes: Setting of true model parameters
import pandas
import math
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from scipy.stats import poisson
from scipy.stats import norm


######################################################################
# Node number: 5
Num_st = 5 #nodes
Num_time = 100 #time points
Num_month_train = 50 #samples
Num_month_test = 50 #samples
t_T = Num_time #last time
t_1 = 0 #initial time
K = 18 #attributes
L = 3 #layers
R = 3 #communities


##True parameter setting

#Tal: community
rTal = np.zeros((Num_st,R))
rTal[0:1,0] = 1
rTal[1:3,1] = 1
rTal[3:,2] = 1

sns.heatmap(rTal)
pyplot.show()



#Theta: correlation among different communities
rTheta = np.ones((R,R))
rTheta[0,1] = 0.3
rTheta[1,0] = 0.3
rTheta[1,2] = 0.3
rTheta[2,1] = 0.3
rTheta[0,2] = 0.1
rTheta[2,0] = 0.1

sns.heatmap(rTheta)
pyplot.show()



#Gamma: adjacency matrix
rGamma = np.zeros((Num_st,Num_st))
for i in range(0,Num_st):
    numi = np.argwhere(rTal[i,:] == 1)[0,0]
    for j in range(0,Num_st):
        numj = np.argwhere(rTal[j,:] == 1)[0,0]
        rGamma[i,j] = np.random.binomial(1,rTheta[numi,numj])



sns.heatmap(rGamma)
pyplot.show()



#Phi: attributes
rPhi = np.ones((R,K)) * 10
rPhi[0,:6] = 20
rPhi[1,6:12] = 20
rPhi[2,12:K] = 20

sns.heatmap(rPhi)
pyplot.show()



#Mu: background rate
rMu = np.zeros((Num_st,t_T-t_1,L))
for i in range(0,Num_st):
    time = np.array([i for i in range(t_1,t_T)])
    mu0 = np.sin(time/(t_T-t_1) *np.pi * 2)
    mu1 = 15
    mu2 = 20
    mu3 = 25
    rMu[i,:,0] = mu1
    rMu[i,:,1] = mu2
    rMu[i,:,2] = mu3



#Alpha: influence exciting term
V = np.zeros((Num_st,Num_st))
rAlpha0 = np.zeros((Num_st,Num_st,L))
for i in range(0,Num_st):
    for j in range(0,Num_st):
        if abs(i-j) > 0:
            vi = np.random.rand()
            vj = np.random.rand()
        else:
            vi = 0
            vj = 0
        V[i,j] = (vi-vj)**2
        rAlpha0[i,j,0] = 2.2 * np.exp(-0.55 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,1] = 2 * np.exp(-0.5 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,2] = 1.8 * np.exp(-0.45 * 2 * (vi-vj)**2) * 1e-1 * 50


rAlpha = rAlpha0 * np.tile(np.reshape(rGamma,(Num_st,Num_st,1)),(1,1,L)) * 4

pyplot.subplot(2,2,1)        
sns.heatmap(rAlpha[:,:,0],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,2)        
sns.heatmap(rAlpha[:,:,1],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,3)        
sns.heatmap(rAlpha[:,:,2],linewidths=.1,cmap="YlGnBu")
pyplot.show()


#Parameter saving
np.save('rTal.npy',rTal)
np.save('rTheta.npy',rTheta)
np.save('rGamma.npy',rGamma)
np.save('rPhi.npy',rPhi)
np.save('rMu.npy',rMu)
np.save('rAlpha.npy',rAlpha)



######################################################################
# Node number: 10
Num_st = 10 #nodes
Num_time = 100 #time points
Num_month_train = 50 #samples
Num_month_test = 50 #samples
t_T = Num_time #last time
t_1 = 0 #initial time
K = 18 #attributes
L = 3 #layers
R = 3 #communities


##True parameter setting

#Tal: community
rTal = np.zeros((Num_st,R))
rTal[0:2,0] = 1
rTal[2:6,1] = 1
rTal[6:,2] = 1

sns.heatmap(rTal)
pyplot.show()



#Theta: correlation among different communities
rTheta = np.ones((R,R))
rTheta[0,1] = 0.3
rTheta[1,0] = 0.3
rTheta[1,2] = 0.3
rTheta[2,1] = 0.3
rTheta[0,2] = 0.1
rTheta[2,0] = 0.1

sns.heatmap(rTheta)
pyplot.show()



#Gamma: adjacency matrix
rGamma = np.zeros((Num_st,Num_st))
for i in range(0,Num_st):
    numi = np.argwhere(rTal[i,:] == 1)[0,0]
    for j in range(0,Num_st):
        numj = np.argwhere(rTal[j,:] == 1)[0,0]
        rGamma[i,j] = np.random.binomial(1,rTheta[numi,numj])



sns.heatmap(rGamma)
pyplot.show()



#Phi: attributes
rPhi = np.ones((R,K)) * 10
rPhi[0,:6] = 20
rPhi[1,6:12] = 20
rPhi[2,12:K] = 20

sns.heatmap(rPhi)
pyplot.show()



#Mu: background rate
rMu = np.zeros((Num_st,t_T-t_1,L))
for i in range(0,Num_st):
    time = np.array([i for i in range(t_1,t_T)])
    mu0 = np.sin(time/(t_T-t_1) *np.pi * 2)
    mu1 = 15
    mu2 = 20
    mu3 = 25
    rMu[i,:,0] = mu1
    rMu[i,:,1] = mu2
    rMu[i,:,2] = mu3



#Alpha: influence exciting term
V = np.zeros((Num_st,Num_st))
rAlpha0 = np.zeros((Num_st,Num_st,L))
for i in range(0,Num_st):
    for j in range(0,Num_st):
        if abs(i-j) > 0:
            vi = np.random.rand()
            vj = np.random.rand()
        else:
            vi = 0
            vj = 0
        V[i,j] = (vi-vj)**2
        rAlpha0[i,j,0] = 2.2 * np.exp(-0.55 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,1] = 2 * np.exp(-0.5 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,2] = 1.8 * np.exp(-0.45 * 2 * (vi-vj)**2) * 1e-1 * 50


rAlpha = rAlpha0 * np.tile(np.reshape(rGamma,(Num_st,Num_st,1)),(1,1,L)) * 2

pyplot.subplot(2,2,1)        
sns.heatmap(rAlpha[:,:,0],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,2)        
sns.heatmap(rAlpha[:,:,1],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,3)        
sns.heatmap(rAlpha[:,:,2],linewidths=.1,cmap="YlGnBu")
pyplot.show()


#Parameter saving
np.save('rTal.npy',rTal)
np.save('rTheta.npy',rTheta)
np.save('rGamma.npy',rGamma)
np.save('rPhi.npy',rPhi)
np.save('rMu.npy',rMu)
np.save('rAlpha.npy',rAlpha)



######################################################################
# Node number: 15
Num_st = 15 #nodes
Num_time = 100 #time points
Num_month_train = 50 #samples
Num_month_test = 50 #samples
t_T = Num_time #last time
t_1 = 0 #initial time
K = 18 #attributes
L = 3 #layers
R = 3 #communities


##True parameter setting

#Tal: community
rTal = np.zeros((Num_st,R))
rTal[0:5,0] = 1
rTal[5:10,1] = 1
rTal[10:,2] = 1

sns.heatmap(rTal)
pyplot.show()



#Theta: correlation among different communities
rTheta = np.ones((R,R))
rTheta[0,1] = 0.3
rTheta[1,0] = 0.3
rTheta[1,2] = 0.3
rTheta[2,1] = 0.3
rTheta[0,2] = 0.1
rTheta[2,0] = 0.1

sns.heatmap(rTheta)
pyplot.show()



#Gamma: adjacency matrix
rGamma = np.zeros((Num_st,Num_st))
for i in range(0,Num_st):
    numi = np.argwhere(rTal[i,:] == 1)[0,0]
    for j in range(0,Num_st):
        numj = np.argwhere(rTal[j,:] == 1)[0,0]
        rGamma[i,j] = np.random.binomial(1,rTheta[numi,numj])



sns.heatmap(rGamma)
pyplot.show()



#Phi: attributes
rPhi = np.ones((R,K)) * 10
rPhi[0,:6] = 20
rPhi[1,6:12] = 20
rPhi[2,12:K] = 20

sns.heatmap(rPhi)
pyplot.show()



#Mu: background rate
rMu = np.zeros((Num_st,t_T-t_1,L))
for i in range(0,Num_st):
    time = np.array([i for i in range(t_1,t_T)])
    mu0 = np.sin(time/(t_T-t_1) *np.pi * 2) 
    mu1 = 15
    mu2 = 20
    mu3 = 25
    rMu[i,:,0] = mu1
    rMu[i,:,1] = mu2
    rMu[i,:,2] = mu3



#Alpha: influence exciting term
V = np.zeros((Num_st,Num_st))
rAlpha0 = np.zeros((Num_st,Num_st,L))
for i in range(0,Num_st):
    for j in range(0,Num_st):
        if abs(i-j) > 0:
            vi = np.random.rand()
            vj = np.random.rand()
        else:
            vi = 0
            vj = 0
        V[i,j] = (vi-vj)**2
        rAlpha0[i,j,0] = 2.2 * np.exp(-0.55 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,1] = 2 * np.exp(-0.5 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,2] = 1.8 * np.exp(-0.45 * 2 * (vi-vj)**2) * 1e-1 * 50


rAlpha = rAlpha0 * np.tile(np.reshape(rGamma,(Num_st,Num_st,1)),(1,1,L)) * 20/15

pyplot.subplot(2,2,1)        
sns.heatmap(rAlpha[:,:,0],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,2)        
sns.heatmap(rAlpha[:,:,1],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,3)        
sns.heatmap(rAlpha[:,:,2],linewidths=.1,cmap="YlGnBu")
pyplot.show()


#Parameter saving
np.save('rTal.npy',rTal)
np.save('rTheta.npy',rTheta)
np.save('rGamma.npy',rGamma)
np.save('rPhi.npy',rPhi)
np.save('rMu.npy',rMu)
np.save('rAlpha.npy',rAlpha)



######################################################################
# Node number: 25
Num_st = 25 #nodes
Num_time = 100 #time points
Num_month_train = 50 #samples
Num_month_test = 50 #samples
t_T = Num_time #last time
t_1 = 0 #initial time
K = 18 #attributes
L = 3 #layers
R = 3 #communities


##True parameter setting

#Tal: community
rTal = np.zeros((Num_st,R))
rTal[0:8,0] = 1
rTal[8:16,1] = 1
rTal[16:,2] = 1

sns.heatmap(rTal)
pyplot.show()



#Theta: correlation among different communities
rTheta = np.ones((R,R))
rTheta[0,1] = 0.3
rTheta[1,0] = 0.3
rTheta[1,2] = 0.3
rTheta[2,1] = 0.3
rTheta[0,2] = 0.1
rTheta[2,0] = 0.1

sns.heatmap(rTheta)
pyplot.show()



#Gamma: adjacency matrix
rGamma = np.zeros((Num_st,Num_st))
for i in range(0,Num_st):
    numi = np.argwhere(rTal[i,:] == 1)[0,0]
    for j in range(0,Num_st):
        numj = np.argwhere(rTal[j,:] == 1)[0,0]
        rGamma[i,j] = np.random.binomial(1,rTheta[numi,numj])



sns.heatmap(rGamma)
pyplot.show()



#Phi: attributes
rPhi = np.ones((R,K)) * 10
rPhi[0,:6] = 20
rPhi[1,6:12] = 20
rPhi[2,12:K] = 20

sns.heatmap(rPhi)
pyplot.show()



#Mu: background rate
rMu = np.zeros((Num_st,t_T-t_1,L))
for i in range(0,Num_st):
    time = np.array([i for i in range(t_1,t_T)])
    mu0 = np.sin(time/(t_T-t_1) *np.pi * 2)
    #a1 = 
    #a2 = 
    #a3 = 
    mu1 = 15
    mu2 = 20
    mu3 = 25
    rMu[i,:,0] = mu1
    rMu[i,:,1] = mu2
    rMu[i,:,2] = mu3



#Alpha: influence exciting term
V = np.zeros((Num_st,Num_st))
rAlpha0 = np.zeros((Num_st,Num_st,L))
for i in range(0,Num_st):
    for j in range(0,Num_st):
        if abs(i-j) > 0:
            vi = np.random.rand()
            vj = np.random.rand()
        else:
            vi = 0
            vj = 0
        V[i,j] = (vi-vj)**2
        rAlpha0[i,j,0] = 2.2 * np.exp(-0.55 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,1] = 2 * np.exp(-0.5 * 2 * (vi-vj)**2) * 1e-1 * 50
        rAlpha0[i,j,2] = 1.8 * np.exp(-0.45 * 2 * (vi-vj)**2) * 1e-1 * 50


rAlpha = rAlpha0 * np.tile(np.reshape(rGamma,(Num_st,Num_st,1)),(1,1,L)) * 20/25

pyplot.subplot(2,2,1)        
sns.heatmap(rAlpha[:,:,0],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,2)        
sns.heatmap(rAlpha[:,:,1],linewidths=.1,cmap="YlGnBu")
pyplot.subplot(2,2,3)        
sns.heatmap(rAlpha[:,:,2],linewidths=.1,cmap="YlGnBu")
pyplot.show()


#Parameter saving
np.save('rTal.npy',rTal)
np.save('rTheta.npy',rTheta)
np.save('rGamma.npy',rGamma)
np.save('rPhi.npy',rPhi)
np.save('rMu.npy',rMu)
np.save('rAlpha.npy',rAlpha)
