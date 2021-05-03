from numpy import *
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.io as scio
import csv
import os


def c2ddc_FD(d,c,delta_x):
    # Central differential Format
    [m,n] = c.shape
    ddc = np.zeros([m,n])

    for i in range(2):
        for j in range(2):
            c_diffdiff = np.zeros([m-2,1])
            c_diff = np.zeros([m-1,1])
            d_mid = np.zeros([m-1,1])
            c_diff[:,0] = c[1:,j] - c[0:-1,j]
            d_mid[:,0] = 0.5*(d[1:,i,j] + d[0:-1,i,j])
            c_diffdiff[:,0] = c_diff[1:,0] * d_mid[1:,0] - c_diff[0:-1,0] * d_mid[0:-1,0]
            c_diffdiff[:,0] = c_diffdiff[:,0] / (delta_x**2)
            ddc[1:-1,i] = ddc[1:-1,i] + c_diffdiff[:,0]

    return ddc

def C2diffCoef(C,Rand0,Rand1,Rand2, Rand3, Rand4):
    # Definition of Diffusion Coefficients
    [N_x,N_c] = C.shape
    diffusion_coefficient = np.ones([N_x,N_c,N_c])#*1e-6
    dummy_c= np.ones([N_c,N_c])
    for i in range(N_x):
        diffusion_coefficient[i,:,:] = Rand0
        for j in range(N_c):
            diffusion_coefficient[i,:,:] = diffusion_coefficient[i,:,:]+C[i,j]*Rand1[:,:,j]+(C[i,j]**2)*Rand2[:,:,j]+(C[i,j]**3)*Rand3[:,:,j]+(C[i,j]**4)*Rand4[:,:,j]
    return diffusion_coefficient

def RK4(C,delta_x,delta_t,Rand0,Rand1,Rand2,Rand3,Rand4):
    # Fourth order Runge-Kutta Method
    # 4-stage K1,K2,K3,K4  C = C + (1/6)*delta_t*(K1+2*K2+2*K3+K4)
    [N_x,N_c] = C.shape
    # Diffusion matrix test Type 3:  c dependent
    diffusion_coeffcient = C2diffCoef(C,Rand0,Rand1,Rand2,Rand3,Rand4);
    # compute right-hand side derivative for each component
    ddC = c2ddc_FD(diffusion_coeffcient, C, delta_x)
    # First stage of RK4
    K1 = delta_t*ddC;
    # Second stage of RK4
    diffusion_coeffcient = C2diffCoef(C+0.5*K1,Rand0,Rand1,Rand2,Rand3,Rand4);
    ddC = c2ddc_FD(diffusion_coeffcient, C+0.5*K1, delta_x)
    K2 = delta_t*ddC;
    # Third stage of RK4
    diffusion_coeffcient = C2diffCoef(C+0.5*K2,Rand0,Rand1,Rand2,Rand3,Rand4);
    ddC = c2ddc_FD(diffusion_coeffcient, C+0.5*K2, delta_x)
    K3 = delta_t*ddC;
    # Fourth stage of RK4
    diffusion_coeffcient = C2diffCoef(C+K3,Rand0,Rand1,Rand2,Rand3,Rand4);
    ddC = c2ddc_FD(diffusion_coeffcient, C+K3, delta_x)
    K4 = delta_t*ddC;
    # The finial solution
    K = C + (1/6)*(K1+2*K2+2*K3+K4);
    return K

def Lg(x,r):
    # x is the location, r is the degree
    [m,n] = x.shape
    T = np.zeros([m,n])
    if r == 0:
        T = np.ones([m,n])

    if r == 1:
        T = x

    if r > 1:
        T0 = np.ones([m,n])
        T1 = x
        for i in range(2,r+1):
            T =((2*i-1)*x*T1-(i-1)*T0)/i
            T0 = T1
            T1 = T
            
    return T

def Lgd(x,r):
    # x is location, r is the degree
    [m,n] = x.shape
    if r == 0:
        T = np.zeros([m,n])
        
    if r ==1:
        T = np.ones([m,n])

    if r > 1:
        T1 = np.ones([m,n])
        for i in range(2,r+1):
            T = i*Lg(x,i-1)+x*T1
            T1 = T
            
    return T  

def derivative(C,x):
    [N_x,N_c] =C.shape
    dcdx = np.zeros([N_x,N_c])
    r = 4
    # Degree of the polynomials
    M = np.zeros([r+1,r+1])
    for i in range(N_c):
        for j in range(N_x):
            
            if j == 0:
                C_sample = np.array([[C[j,i]],[C[j+1,i]],[C[j+2,i]],[C[j+3,i]],[C[j+4,i]]])
                x_sample = np.array([[x[j,0]],[x[j+1,0]],[x[j+2,0]],[x[j+3,0]],[x[j+4,0]]])
                for k in range(r+1):
                    M_k = Lg(x_sample,k)
                    M[:,k] = M_k[:,0]
                # compute the Coefficient Matrix
                a = np.dot(linalg.pinv2(M),C_sample)
                for k in range(r+1):
                    x_j = Lgd(x[j:j+1,:],k)
                    dcdx[j,i] = dcdx[j,i] + a[k,0]*x_j[0,0]
                continue

            if j == 1:
                C_sample = np.array([[C[j-1,i]],[C[j,i]],[C[j+1,i]],[C[j+2,i]],[C[j+3,i]]])
                x_sample = np.array([[x[j-1,0]],[x[j,0]],[x[j+1,0]],[x[j+2,0]],[x[j+3,0]]])
                for k in range(r+1):
                    M_k = Lg(x_sample,k)
                    M [:,k] = M_k[:,0]
                    # compute the Coefficient Matrix
                    a = np.dot(linalg.pinv2(M),C_sample)
                for k in range(r+1):
                    x_j = Lgd(x[j:j+1,:],k)
                    dcdx[j,i] = dcdx[j,i] + a[k,0]*x_j[0,0]
                continue

            if j == N_x-2:
                C_sample = np.array([[C[j-3,i]],[C[j-2,i]],[C[j-1,i]],[C[j,i]],[C[j+1,i]]])
                x_sample = np.array([[x[j-3,0]],[x[j-2,0]],[x[j-1,0]],[x[j,0]],[x[j+1,0]]])
                for k in range(r+1):
                    M_k = Lg(x_sample,k)
                    M [:,k] = M_k[:,0]
                    # compute the Coefficient Matrix
                    a = np.dot(linalg.pinv2(M),C_sample)
                for k in range(r+1):
                    x_j = Lgd(x[j:j+1,:],k)
                    dcdx[j,i] = dcdx[j,i] + a[k,0]*x_j[0,0]
                continue
        
            if j == N_x-1:
                C_sample = np.array([[C[j-4,i]],[C[j-3,i]],[C[j-2,i]],[C[j-1,i]],[C[j,i]]])
                x_sample = np.array([[x[j-4,0]],[x[j-3,0]],[x[j-2,0]],[x[j-1,0]],[x[j,0]]])
                for k in range(r+1):
                    M_k = Lg(x_sample,k)
                    M [:,k] = M_k[:,0]
                    # compute the Coefficient Matrix
                    a = np.dot(linalg.pinv2(M),C_sample)
                for k in range(r+1):
                    x_j = Lgd(x[j:j+1,:],k)
                    dcdx[j,i] = dcdx[j,i] + a[k,0]*x_j[0,0]
                continue

            C_sample = np.array([[C[j-2,i]],[C[j-1,i]],[C[j,i]],[C[j+1,i]],[C[j+2,i]]])
            x_sample = np.array([[x[j-2,0]],[x[j-1,0]],[x[j,0]],[x[j+1,0]],[x[j+2,0]]])
            for k in range(r+1):
                M_k = Lg(x_sample,k)
                M [:,k] = M_k[:,0]
                # compute the Coefficient Matrix
                a = np.dot(linalg.pinv2(M),C_sample)
            for k in range(r+1):
                x_j = Lgd(x[j:j+1,:],k)
                dcdx[j,i] = dcdx[j,i] + a[k,0]*x_j[0,0]
    dcdx = dcdx - dcdx[N_x-1,:]       
    return dcdx

def Matano_plane(C,x):
    # xM = int_{CL}^{CR} x dc / (CR - CL)
    # We use Trapezoidal rule
    [N_x,N_c] = C.shape
    xM = np.zeros([N_c,1])
    for i in range(N_c):
        for j in range(N_x-1):
            
            xM[i,0] = xM[i,0] + 0.5*(C[j+1,i] - C[j,i])*(x[j+1,0]+x[j,0])
            
        xM[i,0] = xM[i,0]/(C[-1,i] - C[0,i])
    return xM

def Matano(C,x,T_stop):
    [N_x,N_c] = C.shape
    Int_C = np.zeros([N_x,N_c])
    xM = Matano_plane(C,x)
    for i in range(N_c):
        for j in range(N_x):
            if j == N_x-1:
                Int_C[j,i] = 0
            else:
                for k in range(j,N_x-1):
                    Int_C[j,i] = Int_C[j,i] + 0.5*(C[k+1,i]-C[k,i])*(x[k+1,0]+x[k,0])
                Int_C[j,i] = Int_C[j,i] - xM[i,0]*(C[-1,i]-C[j,i])
                
    Int_C = Int_C/(2*T_stop)
    return Int_C


####################################################################################
## Forward Computing ##
####################################################################################
delta_x = 0.00125
# Spitial step

delta_t = 0.001#delta_x
# Temporal step

x = np.zeros([int(2/delta_x+1),1])
for i in range(x.shape[0]):
    x[i,0] = -1+i*delta_x
# Discretize spitial domain

T_stop = 10
t = np.zeros([int(T_stop/delta_t+1),1])
for i in range(t.shape[0]):
    t[i,0] = 0 + i*delta_t
# Discretize temporal domain

N_x = x.shape[0]
# number of spitial domain
N_t = t.shape[0]
#number of temporal domain
N_c = 2
# number of component

C0 = np.zeros([N_x,N_c])
C0[:,0] = 0.6 * (x[:,0] <= 0)
C0[:,1] = 0.4 * (x[:,0] >=0)

# Generate random coeffcients of diffusion polynomials term
Rand0 =  0.2+0.5*rd.rand(N_c,N_c)
Rand1 =  rd.rand(N_c,N_c,N_c)
Rand2 =  rd.rand(N_c,N_c,N_c)
Rand3 =  rd.rand(N_c,N_c,N_c)
Rand4 =  rd.rand(N_c,N_c,N_c)
dummy_m = np.eye(N_c,N_c)*9+np.ones([N_c,N_c])
# Build Symmetric Matrix
Rand0 = sqrt(Rand0*Rand0.T)*1e-5
Rand0 = Rand0 * dummy_m
for i in range(N_c):  
    Rand1[:,:,i] = sqrt(Rand1[:,:,i]*Rand1[:,:,i].T)*1e-6
    Rand2[:,:,i] = sqrt(Rand2[:,:,i]*Rand2[:,:,i].T)*1e-7
    Rand3[:,:,i] = sqrt(Rand3[:,:,i]*Rand3[:,:,i].T)*1e-8
    Rand4[:,:,i] = sqrt(Rand4[:,:,i]*Rand4[:,:,i].T)*1e-9
# Weights Term
##for i in range(N_c):
    Rand1[:,:,i] = Rand1[:,:,i] * dummy_m
    Rand2[:,:,i] = Rand2[:,:,i] * dummy_m
    Rand3[:,:,i] = Rand3[:,:,i] * dummy_m
    Rand4[:,:,i] = Rand4[:,:,i] * dummy_m



###################################################
# Forward Computing Part
###################################################
tc = 0
# Record time
C = C0
for i in range(2,N_t+1):
    # Forward Simulation
    
    # Fourth-order Runge-Kutta Mehtod
    C = RK4(C,delta_x,delta_t,Rand0,Rand1,Rand2,Rand3,Rand4)
    # Update the Coefficient
    Coef = C2diffCoef(C,Rand0,Rand1,Rand2,Rand3,Rand4)
    # update the time record
    tc = tc + delta_t
    # Update the boundary condition
    C[0,:] = C0[0,:]
    C[-1,:] = C0[-1,:]
    
######################################################################################
## Matano Part ##
######################################################################################
##Noise = rd.normal(0, 0.1, size=(N_x, N_c))
##C = C + Noise
# Add noise data
DuDx = derivative(C,x)
# Using Legendre Polynomials intepolation (Local) to compute the derivative
Int_c = Matano(C,x,T_stop)
# Using Trapezoidal Rule to compute the integral on the Left Side
flux = np.zeros([N_x,N_c])
Coef = C2diffCoef(C,Rand0,Rand1,Rand2,Rand3,Rand4)
# diffusion_coeffcient = C2diffCoef(C,Rand1,Rand2);
for i in range(N_c):
    for j in range(N_c): 
        flux[:,i]= flux[:,i]+Coef[:,i,j]*DuDx[:,j]
######################################################################################
## Save part ##
######################################################################################
# The full data set without deletion

def save_data(x,C,DuDx,Int_c,Coef,Rand0,Rand1,Rand2,Rand3,Rand4):
    [N_x,N_c] = C.shape
##    filename = '../data1/data1_polycase.csv'
    filename = '../data3/data3_polycase7.csv' 
    csvFile=open(filename,'w',newline='')
    try:
        writer=csv.writer(csvFile)
        writer.writerow(('x','c1','c2','dc1','dc2','flux1','flux2'))
        for i in range(N_x):
            writer.writerow((x[i,0],C[i,0],C[i,1],DuDx[i,0],DuDx[i,1],Int_c[i,0],Int_c[i,1]))
    finally:
        csvFile.close()
        
    # Save random coefficient 
##    csvFile=open("../data1/data1_polycase_rand.csv",'w',newline='')
    csvFile=open("../data3/data3_polycase_rand7.csv",'w',newline='')
    try:
        writer=csv.writer(csvFile)
        writer.writerow(('Rand0'))
        for i in range(N_c):
                writer.writerow((Rand0[i,:]))
        writer.writerow(('Rand1'))
        for i in range(N_c):
            for j in range(N_c):
                writer.writerow((Rand1[j,:,i]))
        writer.writerow(('Rand2'))
        for i in range(N_c):
            for j in range(N_c):
                writer.writerow((Rand2[j,:,i]))
        writer.writerow(('Rand3'))
        for i in range(N_c):
            for j in range(N_c):
                writer.writerow((Rand3[j,:,i]))
        writer.writerow(('Rand4'))
        for i in range(N_c):
            for j in range(N_c):
                writer.writerow((Rand4[j,:,i]))
    finally:
        csvFile.close()
    
##    filename = '../data1/data1_polycase_true.csv'
    filename = '../data3/data3_polycase_true7.csv' 
    csvFile=open(filename,'w',newline='')
    try:
        writer=csv.writer(csvFile)
        writer.writerow(('D11','D21','D12','D22'))
        for i in range(N_x):
            writer.writerow((Coef[i,0,0],Coef[i,1,0],Coef[i,0,1],Coef[i,1,1]))
    finally:
        csvFile.close()
    return 0

flag = save_data(x,C,DuDx,Int_c,Coef,Rand0,Rand1,Rand2,Rand3,Rand4)

######################################################################################    
## Plot Part ##
######################################################################################    
fig, ax = plt.subplots()
plt.plot(x[:,0],C[:,0],label = 'C1',linewidth = 4.0, linestyle = '-')
plt.plot(x[:,0],C[:,1],label = 'C2',linewidth = 4.0, linestyle = '-')
ax.set_xlabel('Distance')
ax.set_ylabel('Component')
ax.set_title('The Diffusion Simulation')
ax.legend()
plt.show()
##########        
fig1, ax1 = plt.subplots()
plt.plot(abs(Int_c[:,1]-flux[:,1]),label = 'Matano_error',linewidth = 4.0, linestyle = '-')
ax1.set_xlabel('Distance')
ax1.set_ylabel('IntC-flux')
ax1.set_title('Error of Matano method')
ax1.legend()
plt.show()
########
##fig2, ax2 = plt.subplots()
##plt.plot(abs(Coef_poly[:,0,0]-Coef[:,0,0],),label = 'D11',linewidth = 4.0, linestyle = '-')
##plt.plot(abs(Coef_poly[:,1,1]-Coef[:,1,1],),label = 'D22',linewidth = 4.0, linestyle = '-')
##plt.plot(abs(Coef_poly[:,2,2]-Coef[:,2,2],),label = 'D33',linewidth = 4.0, linestyle = '-')
##ax2.set_xlabel('Distance')
##ax2.set_ylabel('Diffusion Coefficient')
##ax2.set_title('Error of polynomials approaching')
##ax2.legend()
##plt.show()
