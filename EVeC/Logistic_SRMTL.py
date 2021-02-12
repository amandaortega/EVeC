# Logistic_SRMTL
# Sparse Structure-Regularized Learning with Logistic Loss.
# Adapted from the MALSAR package by Amanda O. C. Ayres in Jan 2021
# Updated to include multiple output problems in Jan 2021

# OBJECTIVE
# argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
#            + rho1 * norm(W*R, 'fro')^2 + rho2 * \|W\|_1}

# R encodes the structure relationship
# 1)Structure order is given by using [s12 -s12 0 ...; 0 s23 -s23 ...; ...]
# 2)Ridge penalty term by setting: R = eye(t)
# 3)All related regularized: R = eye (t) - ones (t) / t

# INPUT
#    X: {n * d} * t - input matrix
#    Y: {n * L} * t - output matrix
#    n: number of samples
#    d: input dimension
#    L: output dimenion
#    t: number of tasks
#    R: t * no. connections - regularization structure
#    rho1: structure regularization parameter
#    rho2: sparsity controlling parameter

# OUTPUT
#    W: model: {d * L} * t
#    C: model: L * t
#    funcVal: function value vector.

# RELATED PAPERS

# [1] Evgeniou, T. and Pontil, M. Regularized multi-task learning, KDD 2004
# [2] Zhou, J. Technical Report. http://www.public.asu.edu/~jzhou29/Software/SRMTL/CrisisEventProjectReport.pdf

# RELATED FUNCTIONS
# Least_SRMTL, init_opts

import numpy as np

class Logistic_SRMTL(object):
    # Default values
    DEFAULT_MAX_ITERATION = 1000
    DEFAULT_TOLERANCE     = 1e-4
    DEFAULT_TERMINATION_COND = 1

    def __init__(self, rho_1=0, rho_2=0, rho_3=0):
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.rho_3 = rho_3   
        self.funcVal = None
        self.W = None
        self.C0 = None
    
    def funVal_eval(self, X, Y, W):
        funcVal = 0

        for i in range(self.t):
            funcVal = funcVal + 0.5 * np.linalg.norm(Y[i] - X[i].T @ W[:, :, i]) ** 2

        if self.R is None:
            return funcVal + self.rho_3 * np.linalg.norm(W) ** 2
        return funcVal + self.rho_1 * np.linalg.norm(W @ self.R) ** 2 + self.rho_3 * np.linalg.norm(W) ** 2

    def gradVal_eval(self, X, XY, W):
        grad_W = np.zeros((XY[0].shape[0], XY[0].shape[1], self.t))

        for t_ii in range(self.t):
            XWi = X[t_ii].T @ W[:,:,t_ii]
            XTXWi = X[t_ii] @ XWi
            grad_W[:, :, t_ii] = XTXWi - XY[t_ii]

        if self.RRt is None:
            return grad_W + self.rho_3 * 2 * W
        return grad_W + self.rho_1 * 2 *  W @ self.RRt + self.rho_3 * 2 * W

    # Calculates argmin_z = \|z-v\|_2^2 + beta \|z\|_1 
    # z: solution, l1_comp_val: value of l1 component (\|z\|_1)
    @staticmethod
    def l1_projection(v, beta):        
        z = np.zeros(v.shape)
        vp = v - beta / 2
        z[np.where(v > beta / 2)]  = vp[np.where(v > beta / 2)]
        vn = v + beta / 2
        z[np.where(v < - beta / 2)] = vn[np.where(v < - beta / 2)]
        
        l1_comp_val = np.sum(np.abs(z))

        return [z, l1_comp_val]

    def multi_transpose(self, X):
        for i in range(self.t):
            X[i] = np.insert(X[i], 0, 1, axis=1).T
        
        return X

    def set_RRt(self, R):
        # precomputation        
        if R is None:
            self.RRt = None
        else:
            self.RRt = R @ R.T
        self.R = R

    def train(self, X, Y, init_theta=2):
        self.funcVal = list()
        X = X.copy()

        if init_theta == 1:
            W0 = self.W
            C0 = self.C0
        else:
            self.t = len(X)
            W0 = np.zeros((X[0].shape[1] + 1, Y[0].shape[1], self.t))
        
        X = self.multi_transpose(X)

        # initialize a starting point
        C0_prep = np.zeros(Y[0].shape[1], self.t)

        for t_idx in range(self.t):
            m1 = np.sum(Y[t_idx], axis=0)
            m2 = Y[t_idx].shape[0] - m1
            # ToDo: consider the zero and inf cases
            C0_prep[:, t_idx] = log(m1/m2)
        
        # this flag tests whether the gradient step only changes a little
        bFlag = 0 

        Wz= W0
        Wz_old = W0

        t = 1
        t_old = 0


        gamma = 1
        gamma_inc = 2

        for iter_ in range(Least_SRMTL.DEFAULT_MAX_ITERATION):
            alpha = (t_old - 1) / t
            Ws = (1 + alpha) * Wz - alpha * Wz_old

            # compute the function value and gradients of the search point
            gWs = self.gradVal_eval(X, XY, Ws)
            Fs = self.funVal_eval(X, Y, Ws)

            while True:
                [Wzp, l1c_wzp] = self.l1_projection(Ws - gWs / gamma, 2 * self.rho_2 / gamma)
                Fzp = self.funVal_eval(X, Y, Wzp)
                
                delta_Wzp = Wzp - Ws
                r_sum = np.linalg.norm(delta_Wzp) ** 2

                Fzp_gamma = Fs + np.sum(np.trace(np.matmul(np.moveaxis(delta_Wzp, -1, 0), np.moveaxis(np.moveaxis(gWs, -1, 0), -1, 1)), axis1=1, axis2=2)) + gamma / 2 * np.linalg.norm(delta_Wzp) ** 2
                
                if r_sum <= 1e-20:
                    # this shows that the gradient step makes little improvement
                    bFlag = 1  
                    break
                
                if Fzp <= Fzp_gamma:
                    break
                else:
                    gamma = gamma * gamma_inc
            
            Wz_old = Wz
            Wz = Wzp
            
            self.funcVal.append(Fzp + self.rho_2 * l1c_wzp)

            if bFlag:
                # The program terminates as the gradient step changes the solution very small
                break

            # test stop condition
            if iter_ >= 2:
                if abs(self.funcVal[-1] - self.funcVal[-2]) <= Least_SRMTL.DEFAULT_TOLERANCE * self.funcVal[-2]:
                    break

            t_old = t
            t = 0.5 * (1 + (1 + 4 * t ** 2) ** 0.5)
        
        self.W = Wzp

        return ([self.W[:, :, t] for t in range(self.W.shape[2])], self.C)

    # function value evaluation for each task
    def unit_funcVal_eval(W, C, X, Y):
        m = length(y)
        weight = ones(m, 1)/m
        aa = -y.*(x'*w + c)
        bb = max( aa, 0)
        funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb ) 