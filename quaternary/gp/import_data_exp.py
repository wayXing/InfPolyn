from numpy import *
import numpy as np
import csv

# Automatic Import data computed by simulation
def import_Matano(N_c):
  #N_c : the number of component
  with open('../data2/data2_exp.csv')as f:
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
            DcDx[i,j] = float(f_csv[i+1][4+j])

    for i in range(N_x):
        for j in range(N_c):
            Int_c[i,j] = float(f_csv[i+1][7+j])
            
    f.close()
    
    return x,C,DcDx,Int_c

# Import ture diffusion Coefficients
def import_true_diff(N_c):
  with open('../data2/data2_exp_true.csv')as f:
    f_csv = csv.reader(f)
    f_csv = list(f_csv)
  # Number of samples
  N_x = len(f_csv)-1
  D = np.zeros([N_x,N_c,N_c])
  # initilize Diffusion Matrix
  for i in range(N_x):
    for j in range(N_c):
      for k in range(N_c):
        D[i,j,k] = float(f_csv[i+1][3*j+k])
        
  f.close()
  
  return D

#Import Polynomials fitting Results
def import_poly_diff(r,N_c):
  #r is the poly degree, N_c is the number of component
  r = str(r)
  with open('../data2/data2_exp_poly'+r+'.csv')as f:
    f_csv = csv.reader(f)
    f_csv = list(f_csv)
  # Number of samples  
  N_x = len(f_csv)-1
  D_poly = np.zeros([N_x,N_c,N_c])
  # initial Diffusion Matrix
  for i in range(N_x):
    for j in range(N_c):
      for k in range(N_c):
        D_poly[i,j,k] = float(f_csv[i+1][3*j+k])

  f.close()
        
  return D_poly


