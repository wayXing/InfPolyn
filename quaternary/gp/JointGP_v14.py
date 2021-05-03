# GRP torch model using nn.module
# mixing of n ghost GP (inducing point GP)

# TODO: add regurzlizer to ghost point
# TODO: add prior for the mixing weight
# TODO: add normalize to GP


# %%
# for MAC OS to avoid error: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

import math as math


# from torch.autograd import Variable
JITTER = 1e-6
PI = 3.1415
N_GHOST = 10

##from ghost_cigp_v02 import ghost_cigpr
from ghost_cigp_v03 import ghost_cigpr
class linear_jointGP(nn.Module):
    # linearly combine Nc ghost_cigpr_i * dc_i
    def __init__(self, C):
        super(linear_jointGP, self).__init__()
        # this model solve for GP0(C)*dc[:,0] + GP1(C)*dc[:,1] + ... = y
        # C and dC are N x Nc matrix
        # NGHOST_DECAY = 2

        C = torch.tensor(C)
        N, Nc = C.shape
        # N_ghost = math.ceil(N / 10)
        N_ghost = math.ceil(N_GHOST)

        self.sub_models = []
        for i in range(Nc):
            self.sub_models.append( ghost_cigpr(C, N_ghost) )

        # the weight for each component (GP)
        self.a = nn.Parameter( torch.zeros(Nc) )
        self.a.data[0] = 1.

        # self.a = nn.Parameter( torch.ones(Nc) )
        # self.a.data[1] = 0.

        # self.a[0].data = torch.tensor(1.)
        # self.a.copy_(torch.tensor([1.,0]))

        for i, obj in enumerate(self.sub_models):
            print(i, obj, sep=' ')
        # for obj in self.sub_models:
        #     print(obj.name, obj.roll, sep=' ')

        # xtr = torch.hstack((c1, c2))
        # self.model1 = ghost_cigpr(xtr, math.ceil(N/4) )
        # self.model2 = ghost_cigpr(xtr, math.ceil(N/2) )
        #
        # self.a1 = nn.Parameter(torch.ones(1))
        # self.a2 = nn.Parameter(torch.zeros(1))
        # self.log_a1 = nn.Parameter(torch.zeros(1))
        # self.log_a2 = nn.Parameter(torch.ones(1)) * -100

    # get the final output prediction
    def forward(self, C, dC):
        ypred = 0
        yvar = 0
        for i, obj in enumerate(self.sub_models):
            # print(i, obj, sep=' ')
            ypred1, yvar1 = obj.forward(C)
            ypred = ypred + self.a[i] * ypred1 * dC[:,i:i+1]
            yvar = yvar + self.a[i]**2 * yvar1 * dC[:,i:i+1]**2

        return ypred, yvar

    # get prediction for each component
    def forward_eachGP(self, C):
        ypred = []
        yvar = []
        for i, obj in enumerate(self.sub_models):
            # print(i, obj, sep=' ')
            ypred1, yvar1 = obj.forward(C)
            ypred.append(ypred1 * self.a[i])
            yvar.append(yvar1 * self.a[i]**2)

        ypred = torch.hstack(ypred)
        yvar = torch.hstack(yvar)

        # ypred[:,i] = self.a[i] * ypred1
        # yvar[:,i] = self.a[i]**2 * yvar1
        return ypred, yvar


    def train_adam(self, C, dC, intC, niteration=10, lr=0.001):
        # adam optimizer
        # uncommont the following to enable

        # optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        params = []
        for i, obj in enumerate(self.sub_models):
            # print(i, obj, sep=' ')
            params = params + list(obj.parameters())

        params = params + list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        for i in range(niteration):
            optimizer.zero_grad()

            ypred, yvar = self.forward(C, dC)
            ypred1,yvar1 = self.forward_eachGP(C)
            # prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.t().squeeze(), yvar.squeeze().diag_embed() )
            # loss_prediction = -prob.log_prob(Y_truth.t().squeeze())
            prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.flatten().float(),
                                                                              yvar.flatten().diag_embed().float())
            loss_prediction = -prob.log_prob(intC.flatten().float())

            # prediction loss
            loss = loss_prediction

            # prior for weight
            # first weight has mean 1 and others 0
            for i, obj in enumerate(self.sub_models):
                if i == 0:
##                    loss = loss - torch.distributions.Laplace(1,0.1).log_prob(self.a[i])+torch.exp(- 1e3*torch.min(ypred1[:,i].flatten().float() ) )# 1 0.1
                    loss = loss - torch.distributions.Laplace(1,0.1).log_prob(self.a[i])# 1 0.1
                else:
##                    loss = loss - torch.distributions.Laplace(0.1,0.01).log_prob(self.a[i])+torch.exp(- 1e3*torch.min(ypred1[:,i].flatten().float() )  )# 0.1 0.01
                    loss = loss - torch.distributions.Laplace(0.1,0.01).log_prob(self.a[i])# 0.1 0.01

            # prior for ghost point
            # prob_ghost1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(40)* 0.001, torch.eye(40) * 0.001)
            # loss = loss - prob_ghost1.log_prob(self.model1.Y.flatten().float())
            #
            # prob_ghost2 = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(40), torch.eye(40) * 0.001)
            # loss = loss - prob_ghost2.log_prob(self.model2.Y.flatten().float()) \
            #             - prob_ghost2.log_prob(self.model3.Y.flatten().float())
            #
            for i, obj in enumerate(self.sub_models):
                loss = loss + obj.nll_prior()

            # loss = loss - torch.distributions.normal(0.0, 0.1).log_prob(self.Y)
            loss.backward()
            optimizer.step()
            print('loss_total:', loss.item() )

# %%
if __name__ == "__main__":
    print('testing')
    print(torch.__version__)

    c1 = torch.linspace(0.1, 0.9, 100).view(-1, 1)
    c2 = torch.linspace(0.9, 0.1, 100).view(-1, 1)

    dc1 = torch.ones(100,1) * 0.001
    dc2 = torch.ones(100, 1) * -0.001

    d11 = 0.1 + c1 * 0.1 + c1**2 * 0.2 + c2 * 0.01 + c2**2 * 0.02
    d12 = 0.01 + c1 * 0.01 + c1 ** 2 * 0.02 + c2 * 0.001 + c2 ** 2 * 0.002

    flux = dc1 * d11 + dc2 * d12

    # xte = torch.linspace(0, 6, 100).view(-1, 1)
    # yte = torch.sin(xte) * 0.5 + torch.exp(-xte)
    #
    # xtr = torch.rand(32, 1) * 6
    # ytr = torch.sin(xtr) * 0.5 + torch.exp(-xtr) + torch.rand(32, 1) * 0.1
    plt.plot(dc1 * d11, 'g-')
    plt.plot(dc2 * d12, 'y-')

    plt.plot(d11, 'g--')
    plt.plot(d12, 'y--')

    plt.plot(flux, 'b-')
    plt.show()

    # %%
    C = torch.hstack((c1, c2))
    dC = torch.hstack((dc1, dc2))

    model = linear_jointGP(C)
    model.train_adam(C, dC, flux, niteration=500, lr=0.005)

    # %%
    # model = cigpr(xtr, ytr)
    # model.train_adam(100)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)  # lr is very important, lr>0.1 lead to failure
    # # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.ghost_train_adam(xtr, ytr, 100)

    with torch.no_grad():

        xte = torch.hstack((c1, c2))
        ypred1, yvar2 = model.sub_models[0].forward(xte)
        ypred2, yvar2 = model.sub_models[1].forward(xte)

        ypred1 = model.a[0] * ypred1
        ypred2 = model.a[1] * ypred2

    # plt.plot(xte, ypred.detach(), 'b-')
    # plt.plot(xte, ypred2.detach(), 'g-')
    # plt.plot(xte, ypred.detach() + ypred2.detach(), 'r-')
    # plt.plot(xtr, ytr, 'ko')
    # plt.plot(model1.X.data, model1.Y.data, 'bo')
    # plt.plot(model2.X.data, model2.Y.data, 'go')
    # plt.show()


    # plt.plot(dc1 * d11, 'g-')
    # plt.plot(dc2 * d12, 'y-')

    plt.plot(d11, 'g-')
    plt.plot(d12, 'y-')

    plt.plot(ypred1.detach(), 'g-.')
    plt.plot(ypred2.detach(), 'y-.')
    # plt.plot(flux, 'b-')
    plt.show()

    flux_pred = dc1 * ypred1 + dc2 * ypred2
    plt.figure(2)
    plt.plot(flux, 'b-x')
    plt.plot(flux_pred, 'r-')
    # plt.show()


