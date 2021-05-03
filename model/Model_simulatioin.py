from numpy import *
import numpy as np
from scipy import linalg
import scipy.io as scio
import csv
import sys
sys.path.append("..")
from utils.Polynomial_fitting import polynomials
from utils.InfPolyn import Infpolyn
from utils.L1_optimal import L1
from utils.L2_optimal import L2

'''
Run Polynomials fitting, InfPoly, L1 optimization, L2 optimization method
'''

for i  in range(1,3):
    # ternary components C1 and C2
    # infPolyn model
    Infpolyn(i)
    # L1 optimizer
    L1(i)
    # L2 optimizer
    L2(i)

for r in range(3,5):
    # Degree for polynomials fitting r = 3,4(recommended)
    polynomials(r)
