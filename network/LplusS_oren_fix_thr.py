import torch
import torch.nn as nn
from torch.autograd import Variable
from math import sqrt
import numpy as np
import time


def to_var(X):
    if torch.cuda.is_available():
        X = X.cuda()
    return Variable(X)
    
def to_var_fix(X):
    if torch.cuda.is_available():
        X = X.cuda()
    return Variable(X, requires_grad=False)

# Define a single ISTA cell - inherits from nn.Module
class ISTA(nn.Module):
    def __init__(self, M, N, Iters):
        # Define functions of self
        super(ISTA, self).__init__()  # Init - inheritance
        self.M = M
        self.N = N
        self.Iters = Iters
        self.P_1 = nn.Linear(N, N, bias=True)  # I - (tk)E^H*E: E is M X M
        self.P_2 = nn.Linear(N, N, bias=True)  # (tk)E^H*E: E is M X M
        self.P_3 = nn.Linear(N, N, bias=True)  # tk)E^H  -  For multiplying the measurements in the beginning
        self.P_4 = nn.Linear(N, N, bias=True)  # I - (tk)E^H*E: E is M X M
        self.P_5 = nn.Linear(N, N, bias=True)  # (tk)E^H*E: E is M X M
        self.P_6 = nn.Linear(N, N, bias=True)  # tk)E^H  -  For multiplying the measurements in the beginning
        self.lamb_S = nn.Linear(1, 1, bias=False)  # Thresholding
        self.lamb_S.weight.data = torch.ones(1, 1) * 0.001 # what we used in FISTA
        self.lamb_L = nn.Linear(1, 1, bias=False) # Thresholding
        self.lamb_L.weight.data = torch.ones(1, 1) * 0.02 # what we used in FISTA
        self.OneVec_S = to_var_fix(torch.ones(1, 1))  # Work-around to train the thresholding parameter
        self.OneVec_L = to_var_fix(torch.ones(1, 1))  # Work-around to train the thresholding parameter
        #self.drp = nn.Dropout(p=0.5) # Dropout

    # Unfold FISTA for several iterations
    def forward(self, D):
        # Initialize unfolded network
        
        if torch.cuda.is_available():
            if self.lamb_L.weight.data < torch.ones(1, 1).cuda() * 0.0001:
                self.lamb_L.weight.data = torch.ones(1, 1).cuda() * 0.02  # what we used in FISTA
        else:
            if self.lamb_L.weight.data < torch.ones(1, 1)* 0.0001:
                self.lamb_L.weight.data = torch.ones(1, 1)* 0.02  # what we used in FISTA

        L, S = self.single_cell(self.init_L_S(), self.init_L_S(), D)

        # Add layers
        for ii in range(self.Iters):
            L, S = self.single_cell(L, S, D)
        return L, S

    # Initialize iterations by the zero matrix
    def init_L_S(self):
        return to_var(torch.zeros(self.M, self.N))


    def threshold(self, g, Thr):
        s = 1 / 1000
        signs = torch.sign(g)
        mag = signs * 0.5 * (torch.abs(g) - Thr - s + ((Thr + s - torch.abs(g)) ** 2 + 4 * s * torch.abs(g)) ** 0.5)
        return mag

    def threshold_2(self, g, lamb):
        # x - torch Variable!
        signs = torch.sign(g)
        mag = torch.max(torch.abs(g)-lamb, to_var(torch.zeros(1,g.size(1))))
        return mag*signs


    # mixed l_1 l_2 thresholding
    def mix_soft(self, X, lamb):
        # calculate the l_2 norm for each row
        X_nrm = torch.sqrt(torch.sum(torch.abs(X) ** 2, 1))
        tmp = torch.max(to_var(torch.zeros(X.size(0))), (1 - lamb / X_nrm)).view(-1, 1)
        tmp_mat = tmp.expand(X.size())
        Y = X * tmp_mat
        return Y


    # Forward application of the ISTA cell
    def single_cell(self, L, S, D):
        # L, S - previous iterations
        # D - measurements (after multiplication with P_3)
        Lamb_L = 0.02
        Lamb_S = 0.001
        #L = self.SVT(torch.add(torch.add(self.drp(self.P_3(D)), self.drp(self.P_1(L))), self.drp(self.P_2(S))), Lamb_L)
        L = self.SVT(torch.add(torch.add(self.P_3(D), self.P_1(L)), self.P_2(S)), Lamb_L)
        
        # S = self.threshold(torch.add(torch.add(D, self.P_2(L)), self.P_1(S)), self.lamb_S(self.OneVec_S))
        #S = self.mix_soft(torch.add(torch.add(self.drp(self.P_6(D)), self.drp(self.P_4(L))), self.drp(self.P_5(S))), Lamb_S)
        S = self.mix_soft(torch.add(torch.add(self.P_6(D), self.P_4(L)), self.P_5(S)), Lamb_S)    

        return L, S


    def SVT(self, X, lamb):
        # tic = time.time()
        U, S, V = torch.svd(X)
        # toc = time.time()
        # print(toc-tic)
        tmp = to_var(torch.zeros(V.size()))
        thresholded_vals = self.threshold_2(torch.diag(S), lamb)
        di = np.diag_indices(S.size(0))
        tmp[di] = torch.diag(thresholded_vals)
        Y = torch.matmul(U, torch.matmul(tmp, V.t()))

        return Y

    def view_thresh(self):
        return self.lamb_L(to_var(torch.ones(1, 1))), self.lamb_S(to_var(torch.ones(1, 1)))


"""
    Test the LISTA code
"""

# # Test the class
# M = 15;
# N = 20;
# ista = ISTA(M, N);
# #D    = Variable(torch.randn(1, M)); # Cast X as a pDTorch Variable
# #z    = ista.measurement(D);







