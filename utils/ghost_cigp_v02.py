# GRP torch model using nn.module
# v02: add scale and bias, equivalent to normalize. A stabler version.
# %%
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from torch.autograd import Variable
JITTER = 1e-3

class ghost_cigpr(nn.Module):
    def __init__(self, X, N_ghost, ymean=0, ystd=1.):
        super(ghost_cigpr, self).__init__()
        # self.X = X
        # self.Y = Y
        self.log_beta = nn.Parameter(torch.zeros(1))
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))
        self.log_scale = nn.Parameter(torch.zeros(1))

        self.const_mean = nn.Parameter(torch.ones(1) * ymean)
        self.gain = nn.Parameter(torch.ones(1) * ystd )

        # self.N_ghost = round(X.size(0) / 4)
        # options1: use parameterized X
        self.X = nn.Parameter(X[torch.randperm(X.size(0))[0:N_ghost], :])
        # options2: use fixed X
        # self.X = X[torch.randperm(X.size(0))[0:N_ghost], :]

        # self.Y = nn.Parameter(torch.randn(N_ghost, 1))
        self.Y = nn.Parameter(torch.zeros(N_ghost, 1))

    def K_cross(self, X, X2):
        length_scale = torch.exp(self.log_length_scale).view(1, -1)

        # n = X.size(0)
        # aa = length_scale.expand(X.size(0),1)

        X = X / length_scale.expand(X.size(0), length_scale.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), length_scale.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)
        # K_norm2 = torch.reshape(X_norm2,)
        # x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        # K = -2.0 * torch.mm(X,torch.transpose(X2)) + X_norm2 + torch.transpose(X2_norm2) #the distance matrix
        # K = -2.0 * torch.mm(X, X2.t() ) + X_norm2 + X2_norm2.t() #the distance matrix
        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        # K = -1.0 * torch.exp(self.log_length_scale) * K
        K = self.log_scale.exp() * torch.exp(-K)
        return K

    def forward(self, Xte):
        # with torch.no_grad():
        n_test = Xte.size(0)
        # Sigma = self.K_cross(self.X, self.X) + torch.exp(self.log_beta)^(-1) * torch.eye(self.X.size(0)) + JITTER * torch.eye(self.X.size(0))
        Sigma = self.K_cross(self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))
        kx = self.K_cross(Xte, self.X)

        # mean = torch.mm(kx, torch.solve(Sigma, self.Y).solution)
        # var_diag = torch.exp(self.log_beta).pow(-1).expand(n_test,1) - torch.mm(kx, torch.solve(Sigma, kx.t)).diag().view(-1,1)

        # option1: direct method
        # mean = kx @ torch.lstsq(self.Y, Sigma).solution
        # var_diag = self.log_beta.pow(-1).expand(n_test,1) - (kx@torch.lstsq(kx.t(), Sigma).solution).diag().view(-1,1)

        # option2: via cholesky decompositon
        L = torch.cholesky(Sigma)
        mean = kx @ torch.cholesky_solve(self.Y, L)
        alpha = L.inverse() @ kx.t()

        var_diag = self.log_scale.exp().expand(n_test, 1) - (alpha.t() @ alpha).diag().view(-1, 1)
        var_diag = var_diag + self.log_beta.exp().pow(-1)
        # var_diag = self.log_scale.exp().expand(n_test,1) - (alpha.t()**2).sum(0).view(-1,1)

        mean = mean * self.gain + self.const_mean
        var_diag = var_diag * self.gain**2

        return mean, var_diag

    def negative_log_likelihood(self):
        # x_dimension = Xte.size(1)
        x_num, x_dimension = self.X.shape
        # a = self.K_cross(self.X, self.X)
        # b = torch.exp(self.log_beta).pow(-1)
        Sigma = self.K_cross(self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))
        # a = torch.lstsq(self.Y, Sigma)
        # b = torch.mm(self.Y.t(), a.solution)
        # c = 0.5 * torch.logdet(Sigma)
        # L = 0.5 * torch.logdet(Sigma) + 0.5 * torch.mm(self.Y.t(), torch.lstsq(self.Y, Sigma).solution) # torch.lstsq will not privide gradient
        # direct method:
        # nll = 0.5 * torch.logdet(Sigma) + 0.5 * self.Y.t()@torch.inverse(Sigma)@self.Y # torch.lstsq will not privide gradient
        # use LU decomposition

        L = torch.cholesky(Sigma)
        # nll = 0.5 * 2 * L.diag().log().sum() + 0.5 * self.Y.t() @ torch.cholesky_solve(self.Y, L)  # grad not implmented for solve :(
        # nll = 0.5 * 2 * L.diag().log().sum() + 0.5 * self.Y.t() @ torch.cholesky_inverse(L) @ self.Y    # grad not implmented for solve :(

        alpha = L.inverse() @ self.Y
        # alpha = torch.cholesky_solve(self.Y, L)

        nll = 0.5 * x_dimension * L.diag().log().sum() + 0.5 * (alpha ** 2).sum()
        # nll = torch.norm(self.Y - self.forward(Xte)[0])
        return nll

    def negative_log_likelihood_givenXY(self, X, Y):
        x_num, x_dimension = self.X.shape
        ypred, yvar = self.forward(X)
        prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.flatten().float(),
                                                                          yvar.flatten().diag_embed().float())
        loss = -prob.log_prob(Y.flatten().float())

        # add prior to the ghost point
        N_ghost = self.Y.shape[0]
        prob_ghost1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(N_ghost) * 1,
                                                                                 torch.eye(N_ghost) * 1)
        loss = loss - prob_ghost1.log_prob(self.Y.flatten().float())
        return loss

    # negative log likelihood of prior (of the ghost points)
    def nll_prior(self):
        N_ghost = self.Y.shape[0]
        prob_ghost1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(N_ghost) * 1,
                                                                                 torch.eye(N_ghost) * 1)
        return -prob_ghost1.log_prob(self.Y.flatten().float())


    def train_adam(self, niteration=10, lr=0.001):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood(self.X)
            loss.backward()
            optimizer.step()
            print('loss_nnl:', loss.item())

    def train_bfgs(self, niteration=10, lr=0.001):
        # LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)  # lr is very important, lr>0.1 lead to failure
        for i in range(niteration):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                # self.update()
                loss = self.negative_log_likelihood(self.X)
                loss.backward()
                print('nll:', loss.item())
                return loss

            # optimizer.zero_grad()
            optimizer.step(closure)
        # print('loss:', loss.item())

    def ghost_train_adam(self, X, Y, niteration=10, lr=0.001):
        # adam optimizer
        # uncommont the following to enable

        # reset const mean and bias
        with torch.no_grad():
            # self.gain.data = Y.std().detach()         %Wrong
            # self.const_mean.data = Y.mean().detach()

            self.gain.copy_(Y.std())
            self.const_mean.copy_(Y.mean())

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for i in range(niteration):
            optimizer.zero_grad()

            ypred, yvar = self.forward(X)
            # prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.t().squeeze(), yvar.squeeze().diag_embed() )
            # loss_prediction = -prob.log_prob(Y_truth.t().squeeze())
            prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.flatten().float(),
                                                                              yvar.flatten().diag_embed().float())
            loss_prediction = -prob.log_prob(Y.flatten().float())

            loss = loss_prediction
            # add the prior loss from ghost point
            loss = loss + self.nll_prior()

            # add prior to the ghost point
            # N_ghost = self.Y.shape[0]
            # prob_ghost1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(N_ghost) * 1, torch.eye(N_ghost) * 1)
            # loss = loss - prob_ghost1.log_prob(self.Y.flatten().float())

            loss.backward(retain_graph=True)
            optimizer.step()
            print('iter',i, ' loss_total:', loss.item() )


# %%


if __name__ == "__main__":
    print('testing')
    print(torch.__version__)
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(32, 1) * 6
    ytr = torch.sin(xtr) + torch.rand(32, 1) * 0.5 + 10
    # plt.plot(xtr, ytr, 'bx')

    # model = cigpr(xtr, ytr)
    # model.train_adam(100)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)  # lr is very important, lr>0.1 lead to failure
    # # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model = ghost_cigpr(xtr, 30)
    # %%
    # model.gain.data = ytr.std().detach()
    # model.const_mean.data = ytr.mean().detach()

    # model.gain.copy_(ytr.std())
    # model.const_mean.copy_(ytr.mean())

    model.ghost_train_adam(xtr, ytr, 1000, lr=0.005)

    with torch.no_grad():
        ypred, yvar = model(xte)

    # plt.plot(xte, ypred.detach(), 'r+')
    plt.errorbar(xte, ypred.detach(), yvar.sqrt().squeeze().detach(),fmt='r-.' ,alpha = 0.2)

    plt.plot(xtr, ytr, 'b+')
    plt.plot(model.X.data, model.Y.data, 'ko')
    plt.show()