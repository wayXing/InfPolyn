# Bayesian linear regression with features and prior
'''
L1_optimizer
Compute the optimal by L1-optimization
'''
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import csv
from utils.import_data import import_Matano, import_true_diff

##################################################
# Torch Module
##################################################

class Bayesian_Gaussian(nn.Module):
    def __init__(self,Input,Output,N_c,r):
        super(Bayesian_Gaussian, self).__init__()
        self.Input = Input
        # Input  : [1*dc1dx,c1*dc1dx,c2*dc1dx,c3*dc1dx,...]
        self.Output = Output
        # Output : should equal to Int_c
        self.N_c = N_c
        self.r = r
        # Number of component
        self.row = self.Input.size(0)
        self.column = self.Input.size(1)
        self.mean1 = torch.ones([self.column,1])
        self.std1 = torch.ones([self.column,1])
        self.mean2 = torch.ones([self.column,1])
        self.std2 = torch.ones([self.column,1])
        for j in range(r*N_c+1):
            if j == 0:
                self.mean1[j,0] = 1
                self.std1[j,0] = 1
            else:
                self.mean1[j,0] = (0.1)**int((j-1)/N_c+1)
                self.std1[j,0] =  (0.1)**int((j-1)/N_c+1)
        for j in range(r*N_c+1):
            if j == 0:
                self.mean2[j,0] = 0.1
                self.std2[j,0] = 0.1
            else:
                self.mean2[j,0] = (0.01)**int((j-1)/N_c+1)
                self.std2[j,0] =  (0.01)**int((j-1)/N_c+1)
                    
        # Parameters for [1,c1,c2,c3,c1^2,c2^2.c3^2]
        self.weights1 = nn.Parameter(torch.normal(mean = self.mean1, std = self.std1))
        self.weights2 = nn.Parameter(torch.normal(mean = self.mean2, std = self.std2))    
    def forward(self,Input1,Input2):
        y =  Input1 @ self.weights1 + Input2 @ self.weights2
        return y
    
def fit(Input1,Input2,Output,N_c,r,epoch):
    
    model = Bayesian_Gaussian(Input1,Output,N_c,r)
    opt = optim.Adam(model.parameters(), lr=0.001, eps=1e-08)
    for i in range(epoch):
        pred = model(Input1,Input2)
        # Prediction of Linear combination
        loss_value = torch.sqrt(torch.sum((pred - Output)**2))/model.row+torch.sqrt(torch.sum(torch.abs(model.weights1) ))/model.column\
                     +torch.sqrt(torch.sum(torch.abs(model.weights2) ) )/model.column
        # The loss function
        loss_value.backward()
        # Backward Alogrithm
        opt.step()
        opt.zero_grad()
        if i % 10 == 0:
            print('-log_probability is ',loss_value,'\n')
    return model.weights1, model.weights2
##############################################################
# Prepare Inputs Data
##############################################################
def L1(icomp):
    '''
    Input: icomp ith component for diffusion equation
    icomp = 1 or 2
    '''
    x,C,DcDx,Int_c = import_Matano(2)
    D = import_true_diff(2)
    # import data
    r = 4
    # the polynomials degree
    N_x,N_c = C.shape
    icomp = icomp-1 #0,1
    # degree of polynomials approximation
    N_ghost = 100
    # temp containor
    D_temp = torch.zeros([N_x,N_c])

    for k in range(10):
        # Random Pick Samples From Component
        index = np.random.permutation(np.arange(int(N_x/2)-50,int(N_x/2)+50))[0:N_ghost]
        C_ghost = C[index,:]
        DcDx_ghost = DcDx[index,:]
        Int_C_ghost = np.zeros([N_ghost,1])
        Int_C_ghost[:,0] = Int_c[index,icomp]
        # Normalize flux value

        # Set ghost points
        N_x = N_ghost

        dummy_C = np.zeros([N_x,r*N_c+1])
        for i in range(1,r+1):
            dummy_C[:,1+(i-1)*N_c:N_c+1+(i-1)*N_c] = np.power(C_ghost,i)
        dummy_C[:,0] = np.ones([N_x]) 

        dummy_C1 = np.zeros([N_x,r*N_c+1])
        dummy_C2 = np.zeros([N_x,r*N_c+1])

        for i in range(r*N_c+1):
            dummy_C1[:,i] = dummy_C[:,i]*DcDx_ghost[:,0]
        for i in range(r*N_c+1):
            dummy_C2[:,i] = dummy_C[:,i]*DcDx_ghost[:,1]
        # Value the coefficients matrix
        Input1 =  torch.from_numpy(dummy_C1)
        Input1 = Input1.float()
        Input2 =  torch.from_numpy(dummy_C2)
        Input2 = Input2.float()
        # Value the flux term
        Output = np.zeros([N_x,1])
        Output[:,0] = Int_C_ghost[:,0]
        Output = torch.from_numpy(Output)
        Output = Output.float()
        Output_std = Output.std()
        Output = Output/Output_std 

        # compute the Weights
        weights1,weights2 = fit(Input1,Input2,Output,N_c,r,2000)

        # Compute the coefficients
        N_x,N_c = C.shape
        # redefine the shape of component
        dummy_C = np.zeros([N_x,r*N_c+1])
        for i in range(1,r+1):
            dummy_C[:,1+(i-1)*N_c:N_c+1+(i-1)*N_c] = np.power(C,i)
        dummy_C[:,0] = np.ones([N_x])
        dummy_C = torch.from_numpy(dummy_C)
        dummy_C = dummy_C.float()

        D_Bayesian = torch.ones([N_x,N_c])
        # Compute the coefficients
        D_Bayesian[:,0] = (dummy_C @ weights1)[:,0]
        D_Bayesian[:,1] = (dummy_C @ weights2)[:,0]
        D_Bayesian = D_Bayesian*Output_std 
        D_temp = D_Bayesian+D_temp
    # average of solution
    D_Bayesian = D_temp/10
    D_Bayesian = D_Bayesian.detach().numpy()    
    csvFile=open('../data/L1_optimal_'+str(icomp+1)+'.csv','w',newline='')
    try:
        writer=csv.writer(csvFile)
        writer.writerow(('D'+str(icomp)+'0','D'+str(icomp)+'1'))
        for j in range(N_x):
            writer.writerow((D_Bayesian[i,0],D_Bayesian[i,1]))
    finally:
        csvFile.close()






