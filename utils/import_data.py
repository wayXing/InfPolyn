from numpy import *
import numpy as np
import csv

# Automatic Import data computed by forward-computing
def import_Matano(N_c):
  #N_c : the number of component
  with open('../data/data_polycase.csv')as f:
    f_csv = csv.reader(f)
    f_csv = list(f_csv)
    # dimension of component
    N_x = len(f_csv)-1
    # grids points of domain
    x = np.zeros([N_x,1])
    C = np.zeros([N_x,N_c])
    DcDx = np.zeros([N_x,N_c])
    Int_c = np.zeros([N_x,N_c])
    
    #  initialize data size
    for i in range(N_x):
        for j in range(N_c):
            C[i,j] = float(f_csv[i+1][1+j])

    for i in range(N_x):
        x[i,0] = float(f_csv[i+1][0])

    for i in range(N_x):
        for j in range(N_c):
            DcDx[i,j] = float(f_csv[i+1][3+j])

    for i in range(N_x):
        for j in range(N_c):
            Int_c[i,j] = float(f_csv[i+1][5+j])
            
    f.close()
    
    return x,C,DcDx,Int_c

# Import ture diffusion Coefficients
def import_true_diff(N_c):
  with open('../data/data_polycase_true.csv')as f:
    f_csv = csv.reader(f)
    f_csv = list(f_csv)
  # Number of samples
  N_x = len(f_csv)-1
  D = np.zeros([N_x,N_c,N_c])
  # initilize Diffusion Matrix
  for i in range(N_x):
    for j in range(N_c):
      for k in range(N_c):
        D[i,j,k] = float(f_csv[i+1][N_c*j+k])
        
  f.close()
  
  return D


