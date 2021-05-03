# v13 experiment with real data using JointGP_v14


from JointGP_v14 import linear_jointGP

import import_data
# v10: combine diagonal fitting as init; ghost-cigp with mean and scale

import torch
# from JointGPx3_v03 import jointModelx3  # working model
# from JointGPx3_v04 import jointModelx3
from import_data import import_Matano, import_true_diff, import_poly_diff #, import_true_flux
import matplotlib.pyplot as plt
import numpy as np

# choosing the ith component iComp
iComp = 2   #1,2

x,C,DuDx,Int_c = import_Matano(2)
##flux = import_true_flux(3)
##D_poly = import_poly_diff(1,3)
D = import_true_diff(2)
# D = D[:,iComp-1,:]

id_tr = list(range(700,900,10))
# id_tr = list(range(0,1600,5))
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

fig, ax = plt.subplots()
plt.plot(DuDx[id_tr,0], label = 'dc1',linewidth = 4.0, linestyle = '-')
plt.plot(DuDx[id_tr,1], label = 'dc2',linewidth = 4.0, linestyle = '-')
plt.plot(intC[id_tr,:], label = 'intC',linewidth = 4.0, linestyle = '-')
ax.set_xlabel('Distance')
ax.set_ylabel('Component')
ax.set_title('The Diffusion Simulation')
ax.legend()
##plt.show()

# %%
d11_DiagMamtano = intC/DuDx[:,0:1]

fig, ax = plt.subplots()

id_stable = list(range(500,1100,1))
# plt.plot(x[:,0], d11_DiagMamtano * intC_std, 'b--', label = 'D11_Matano', linewidth = 4.0,)
plt.plot(x[id_stable,0], d11_DiagMamtano[id_stable,:] * intC_std, 'b--', label = 'D11_Matano', linewidth = 4.0,)

plt.plot(x[:,0], D[id_te,iComp-1,1], 'k-', label = 'D dominant', linewidth = 4.0,)
ax.set_xlabel('Distance')
ax.set_ylabel('Component')
ax.set_title('D')
ax.legend()
##plt.show()


# %%
# model = jointModelx2(c1, c2, dc1, dc2)
# model.train_adam(c1, c2, dc1, dc2, intC , niteration=1000, lr=0.001)
d11_mean = d11_DiagMamtano[id_stable,:] .mean().item()
d11_std = d11_DiagMamtano[id_stable,:] .std().item()

model = linear_jointGP(C[id_tr,:])

# pre-train for dominant diffusion coefficient
# model.sub_models[0].ghost_train_adam(C[id_stable,:], d11_DiagMamtano[id_stable,:], niteration=500, lr=0.01)
model.sub_models[0].ghost_train_adam(C[id_tr,:], d11_DiagMamtano[id_tr,:], niteration=1500, lr=0.001)

# show
with torch.no_grad():
    d1, d1var = model.sub_models[0].forward(C[id_te,:])

fig, ax = plt.subplots()
plt.plot(x[id_tr,0], d11_DiagMamtano[id_tr,:] * intC_std, 'x', label = 'training', linewidth = 4.0,)
plt.plot(x[id_te,0], d1 * intC_std, 'b--', label = 'D11_Matano_GP', linewidth = 4.0,)
plt.plot(x[id_te,0], D[id_te,0,iComp-1], 'k-', label = 'D11_true', linewidth = 4.0,)

# plt.plot(dc2, label = 'C2',linewidth = 4.0, linestyle = '-')
# plt.plot(dc3, label = 'C3',linewidth = 4.0, linestyle = '-')
# plt.plot(intC, label = 'intC',linewidth = 4.0, linestyle = '-')
ax.set_xlabel('Distance')
ax.set_ylabel('Component')
ax.set_title('D11')
ax.legend()
##plt.show()


# model1 = ghost_cigpr(C[id_stable,:], 10, d11_mean, d11_std )
# model1.ghost_train_adam(C[id_stable,:], d11_DiagMamtano[id_stable,:], niteration=100, lr=0.001)

# %%
# model = jointModelx3(c1, c2, c3, dc1, dc2, dc3, d11_mean, d11_std )
model.train_adam(C[id_tr,:], DuDx[id_tr,:], intC[id_tr,:], 8000, lr=0.001)
model.train_adam(C[id_tr,:], DuDx[id_tr,:], intC[id_tr,:], 8000, lr=0.001)

# %%
with torch.no_grad():
    intC_pred, intC_var = model.forward(C[id_te,:], DuDx)
    D_pred, D_var = model.forward_eachGP(C[id_te,:])

    # de normalize
    D_pred = D_pred * intC_std
    D_var = D_var * intC_std**2

    # do each prediction manually
    # ypred = []
    # yvar = []
    # for i, obj in enumerate(model.sub_models):
    #     # print(i, obj, sep=' ')
    #     ypred1, yvar1 = obj.forward(C)
    #     ypred.append(ypred1 * model.a[i])
    #     yvar.append(yvar1 * model.a[i] ** 2)
    #
    # D_pred = torch.hstack(ypred)
    # D_var = torch.hstack(yvar)

# %%
# move the domninant componnet back
# D_pred = torch.hstack((ypred1.detach(),ypred2.detach(),ypred3.detach())) *intC_std
# D_var = torch.hstack((yvar1.detach(),yvar2.detach(),yvar3.detach())) *intC_std**2

# DuDx = np.roll(DuDx, (iComp-1), axis=1)
# D = np.roll(D, (iComp-1), axis=1)
D = import_true_diff(2)
# D = D[:,iComp-1,:]

D_pred = D_pred.detach().numpy()
D_var = D_var.detach().numpy()

# roll back to the original order
D_pred = np.roll(D_pred, (iComp-1), axis=1)
D_var = np.roll(D_var, (iComp-1), axis=1)


##import os
##saveFolder = os.path.basename(__file__).split('.')[0]
# try:
#     os.mkdir(saveFolder)
# finally:
#     pass


for i in range(2):
    fig, ax = plt.subplots()

    labelName = 'D'+str(iComp) + str(i+1)
    plt.plot(x[id_te], D_pred[:,i].data, 'r-.', label=labelName+'predictions', linewidth=3.0)
    plt.errorbar(x[id_te], D_pred[:,i].data, np.sqrt(D_var[:,i].data), fmt='r-.', alpha = 0.02)
    plt.plot(x[id_te], D[id_te, iComp-1, i], 'b-', label=labelName+'truth', linewidth=3.0)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Diffusion Coefficient')
    ax.set_title('Our method')
    ax.legend(loc='upper right')
    # plt.show()

    saveName = '../figure/D_' + str(iComp) + '-' + str(i+1)
    plt.savefig(saveName )



# plt.savefig('model3_D13')

# save the model
# torch.save(model.state_dict(),'Good_Exp_jointGP_v10_data2')
#
# load the model
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

# %%
saveName = '../data1/D_pred_' + str(iComp) + '.csv'

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
