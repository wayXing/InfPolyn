# GRP torch model using nn.module
# mixing of n ghost GP (inducing point GP)

# TODO: add regurzlizer to ghost point
# TODO: add prior for the mixing weight
# TODO: add normalize to GP


# for MAC OS to avoid error: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
import math as math
from utils.ghost_cigp_v02 import ghost_cigpr

# from torch.autograd import Variable
JITTER = 1e-6
PI = 3.1415
N_GHOST = 10


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

        for i, obj in enumerate(self.sub_models):
            print(i, obj, sep=' ')

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
            ypred1, yvar1 = obj.forward(C)
            ypred.append(ypred1 * self.a[i])
            yvar.append(yvar1 * self.a[i]**2)

        ypred = torch.hstack(ypred)
        yvar = torch.hstack(yvar)
        return ypred, yvar


    def train_adam(self, C, dC, intC, niteration=10, lr=0.001):
        # adam optimizer
        # uncommont the following to enable
        params = []
        for i, obj in enumerate(self.sub_models):
            params = params + list(obj.parameters())

        params = params + list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        for i in range(niteration):
            optimizer.zero_grad()

            ypred, yvar = self.forward(C, dC)
            ypred1,yvar1 = self.forward_eachGP(C)
            prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.flatten().float(),
                                                                              yvar.flatten().diag_embed().float())
            loss_prediction = -prob.log_prob(intC.flatten().float())

            # prediction loss
            loss = loss_prediction

            # prior for weight
            # first weight has mean 1 and others 0
            for i, obj in enumerate(self.sub_models):
                if i == 0:
                    loss = loss - torch.distributions.Laplace(1,0.1).log_prob(self.a[i])+torch.exp(- 1e4*torch.min(ypred1[:,i].flatten().float() ) )
                else:
                    loss = loss - torch.distributions.Laplace(0.1,0.01).log_prob(self.a[i])+torch.exp(- 1e4*torch.min(ypred1[:,i].flatten().float() )  )

            for i, obj in enumerate(self.sub_models):
                loss = loss + obj.nll_prior()

            loss.backward()
            optimizer.step()
            print('loss_total:', loss.item() )




