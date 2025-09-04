import pandas
import math
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from scipy.stats import poisson
from scipy.stats import norm
from sklearn.cluster import KMeans
import csv
import pandas
import scipy



##################################################
#Setting of true parameter values
Num_st = 20 #nodes
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
rTal[0:4,0] = 1
rTal[4:11,1] = 1
rTal[11:,2] = 1

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


rAlpha = rAlpha0 * np.tile(np.reshape(rGamma,(Num_st,Num_st,1)),(1,1,L))

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






##################################################
#Initialization of community number attributes
st_info, flow_train, flow_train_CIF, flow_test, flow_test_CIF = data_generating()
R = 3 #community number


#Tal
#Attributes only to classify community
#Tal: tal_ir
km = KMeans(n_clusters=R)
km.fit(st_info)
y_pred1 = km.labels_
tal0 = 1/int(R/2)
Tal = (1-tal0)/(R-1) * np.ones((Num_st,R))
for i in range(0,Num_st):
    Tal[i,y_pred1[i]] = tal0



#Phi
Q = 20 #iterative number
Phi = np.ones((R,K))
Pidr = np.zeros((Num_st,R))
Q_Phi = np.ones((R,K,Q))
Q_Tal = np.zeros((Num_st,R,Q))
for q in range(0,Q):
    for r in range(0,R): 
        for i in range(0,Num_st): 
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
            

#community number
com = np.zeros((Num_st))
for i in range(0,Num_st): 
    com[i] = np.argwhere(hat_tal_ird[i,:] == hat_tal_ird[i,:].max())[0,0]



myfile = open('community number_attributes.csv','w')
with myfile:
    writer = csv.writer(myfile)
    writer.writerows(np.reshape(com,(com.shape[0],1)))


