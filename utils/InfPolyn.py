'''
InfPolyn method:
Computing ternary diffusion coefficients by Infpoly method
'''
# v13 experiment with real data using JointGP_v14
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.JointGP_v14 import linear_jointGP
from utils.import_data import import_Matano, import_true_diff


def Infpolyn(iComp):
    # iComp: choosing the ith component iComp
    # iComp = 1 or 2

    # import flux and component data
    x,C,DuDx,Int_c = import_Matano(2)
    D = import_true_diff(2)

    # Location for 20 samples
    id_tr = list(range(700,900,10))
    id_te = list(range(0,1601,1))

    # move the domninant componnet to the first position as default; linear_jointGP a[0]=1 and a[1,2,...]=0 as initial
    DuDx = np.roll(DuDx, (iComp-1), axis=1)
    D = np.roll(D, (iComp-1), axis=1)

    intC = Int_c[:,iComp-1:iComp]


    # convert to torch data
    C = torch.tensor(C).float()
    DuDx = torch.tensor(DuDx).float()
    intC = torch.tensor(intC).float()

    # normalize with only scale
    intC_mean = intC.mean()
    intC_std = intC.std()
    # intC = (intC - intC_mean) / intC_std
    intC = intC / intC_std

    # %%
    d11_DiagMamtano = intC/DuDx[:,0:1]

    model = linear_jointGP(C[id_tr,:])

    # Pre-train for dominant diffusion coefficient
    model.sub_models[0].ghost_train_adam(C[id_tr,:], d11_DiagMamtano[id_tr,:], niteration=1500, lr=0.002)

    # show
    with torch.no_grad():
        d1, d1var = model.sub_models[0].forward(C[id_te,:])

    # Train InfPolyn on the whole Entries
    model.train_adam(C[id_tr,:], DuDx[id_tr,:], intC[id_tr,:], 3000, lr=0.001)
    model.train_adam(C[id_tr,:], DuDx[id_tr,:], intC[id_tr,:], 5000, lr=0.001)


    with torch.no_grad():
        intC_pred, intC_var = model.forward(C[id_te,:], DuDx)
        D_pred, D_var = model.forward_eachGP(C[id_te,:])

        # de normalize
        D_pred = D_pred * intC_std
        D_var = D_var * intC_std**2


    D = import_true_diff(2)


    D_pred = D_pred.detach().numpy()
    D_var = D_var.detach().numpy()

    # Roll back to the original order
    D_pred = np.roll(D_pred, (iComp-1), axis=1)
    D_var = np.roll(D_var, (iComp-1), axis=1)


    # Save InfPolyn results for component iComp
    saveName = '../data/InfPolyn_' + str(iComp) + '.csv'

    import csv
    N_x = D_pred.shape[0]
    csvFile=open(saveName,'w',newline='')
    try:
        writer=csv.writer(csvFile)
        writer.writerow(('D1','D2','D1 var','D2 var'))
        for i in range(N_x):
            writer.writerow((D_pred[i,0].item(),
                             D_pred[i,1].item(),
                             D_var[i,0].item(),
                             D_var[i,1].item()))
    finally:
        csvFile.close()
