from numpy import *
import numpy as np
from scipy import linalg
import scipy.io as scio
import csv
from utils.import_data import import_Matano, import_true_diff

def poly_fit(r,C1,DuDx1,Int_c1,C2):
    [N_x1,N_c1] = C1.shape
    # Initialize size of component
    M = np.zeros([N_c1*N_x1,(r*N_c1+1)*N_c1*N_c1])
    # Initialize Coefficient Matrix
    dummy_C = np.zeros([N_x1, r*N_c1+1])
    # Build Matrix [1,c,c^2,...]
    Int_C = zeros([N_c1*N_x1,1])
    # Flux term
    for i in range(1,r+1):
        dummy_C[:,1+(i-1)*N_c1:N_c1+1+(i-1)*N_c1] = np.power(C1,i)
    dummy_C[:,0] = np.ones([N_x1])
    for i in range(N_x1):
        Int_C[0+i*2:2+i*2,0] = Int_c1[i,:].T
        # Observation vector
    for i in range(N_x1):
        for j in range(N_c1):
            M[0+i*N_c1,0+j*2*(r*N_c1+1):r*N_c1+1+j*2*(r*N_c1+1)] = dummy_C[i,:]*DuDx1[i,j]
            M[1+i*N_c1,r*N_c1+1+j*2*(r*N_c1+1):2*(r*N_c1+1)+j*2*(r*N_c1+1)] = dummy_C[i,:]*DuDx1[i,j]
    # Matrix of Coefficients
    a,residuals,rank,s = np.linalg.lstsq(M,Int_C,rcond=None)
    # Compute diffusion Coefficients
    [N_x2,N_c2] = C2.shape
    Coef_poly = np.zeros([N_x2,N_c2,N_c2])
     # Initialize Coefficient Matrix
    dummy_C2 = np.zeros([N_x2, r*N_c2+1])
    for i in range(1,r+1):
        dummy_C2[:,1+(i-1)*N_c2:N_c2+1+(i-1)*N_c2] = np.power(C2,i)
    dummy_C2[:,0] = np.ones([N_x2])
    for i in range(N_c2):
        for j in range(N_c2):
            Coef_poly[:,j,i] = np.dot(dummy_C2,a[0+j*(r*N_c2+1)+i*2*(r*N_c2+1):r*N_c2+1+j*(r*N_c2+1)+i*2*(r*N_c2+1)])[:,0]
    return Coef_poly

###############################################
#### Main Part ########
###############################################
def polynomials(r):
    '''
    Input r: degree of polynomial model
    Preference r >= 3
    '''
    # Number of component
    N_c =2

    # import_data
    x,C,DuDx,Int_c  = import_Matano(N_c)
    Coef = import_true_diff(N_c)
    N_x = C.shape[0]

    # Number of samples 
    N_ghost = 100
    # Location of samples
    C_sample        =  C[int(N_x/2)-N_ghost:int(N_x/2)+N_ghost:5,:]
    DuDx_sample =  DuDx[int(N_x/2)-N_ghost:int(N_x/2)+N_ghost:5,:]
    Int_c_sample  =  Int_c[int(N_x/2)-N_ghost:int(N_x/2)+N_ghost:5,:]
    # Computing the coefficinet of polynomials model
    Coef_poly = poly_fit(r,C_sample ,DuDx_sample, Int_c_sample,C)

######################################################
#### Save Part
######################################################

    def Save_data(r,N_c):
        # Save polyfiting data
        # r : approximate degree
        # N_c: number of component
        filename = '../data/polynomials_fitting_degree'+str(r)+'.csv' 
        csvFile=open(filename,'w',newline='')
        try:
            writer=csv.writer(csvFile)
            writer.writerow(('D11','D21','D12','D22'))
            for i in range(N_x):
                writer.writerow((Coef_poly[i,0,0],Coef_poly[i,1,0],Coef_poly[i,0,1],Coef_poly[i,1,1]))
        finally:
            csvFile.close()
        return 0

    flag = Save_data(r,N_c)

