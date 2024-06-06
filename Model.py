import math
import matplotlib.pyplot as plt
from matplotlib import style
import time
import scipy.stats as sps
import pandas as pd
import numpy as np
import datetime
import pickle
import csv
from scipy import linalg

import numpy.linalg as lin
import scipy.linalg as lin2
from arch.univariate import arch_model
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, log_likelihood, empirical_covariance
import cvxpy as cp

def interpolate_return(X):

    n = X.shape[0]
    p = X.shape[1]

    xnav = np.zeros((n, p))
    xnav[:] = np.nan

    for j in range(p):
        ind = np.where(~np.isnan(X[:,j]))[0]
        xnav[ind,j] = np.cumprod(1 + X[ind, j])

    tmp = X[0,:].copy()
    tmp[np.isnan(tmp)] = 0
    XX = pd.DataFrame(xnav).interpolate().fillna(method='bfill').pct_change().values
    XX[0,:] = tmp

    return XX


def interpolate_return_2(X):

    n = X.shape[0]
    p = X.shape[1]
    XX = X.copy()
    ss = np.argsort(np.sum(~np.isnan(XX), axis=0))

    for j in range(p):
        ind1 = np.where(np.isnan(XX[:, j]))[0]
        if ind1.size > 0:
            ind2 = np.where(~np.isnan(XX[:,j]))[0]
            np.random.seed(ss[j])
            XX[ind1,j] = np.random.randn(len(ind1))*np.std(XX[ind2, j])+np.mean(XX[ind2,j])

    return XX


def pav(data):
    '''
    确保每个序列单调递增
    '''

    T, N = data.shape
    v = data.copy()
    for j in range(N):
        v[:,j] = pav_1d(v[:,j])

    return v

def pav_1d(y):
    
    y = np.asarray(y)

    n_samples = len(y)
    v = y
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1

    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last

    return v.reshape((len(v), ))

def direct_kernel(X):

    n, p = X.shape
    X = X - np.mean(X, axis=0)
    sample = X.T.dot(X) / n
    sample = (sample+sample.T)/2
    [lamb, u] = lin2.eigh(sample)

    sort_idx = np.argsort(lamb)
    lamb = lamb[sort_idx]
    u = u[:,sort_idx]

    lamb = lamb[np.max([1,p-n+2])-1:]


    L=np.tile(lamb.reshape((len(lamb),1)),(1,len(lamb)))
    h = n**(-0.35)

    tmp = 4*(L.T**2)*h*h -(L-L.T)**2
    tmp[tmp<0] = 0
    ftilde =np.mean(np.sqrt(tmp)/(2*np.pi*(L.T**2)*h*h),axis =1)


    tmp2 = (L-L.T)**2-4*(L.T**2)*h*h
    tmp2[tmp2<0] = 0
    Hftilde =np.mean((np.sign(L-L.T)*np.sqrt(tmp2)-L+L.T)/(2*np.pi*(L.T**2)*h*h), axis =1)

    if p<=n-1:
        dtilde = lamb/((np.pi*(p/n)*lamb*ftilde)**2 +(1-(p/n)-np.pi*(p/n)*lamb*Hftilde)**2)

    else:
        Hftilde0 =(1-np.sqrt(1-4*h*h))/(2*np.pi*h*h)*np.mean(1/lamb)
        dtilde0 = 1/(np.pi*(p-n+1)/(n-1)*Hftilde0)
        dtilde1 = lamb/((np.pi**2)*(lamb**2)*(ftilde**2+Hftilde**2))
        dtilde =np.hstack((dtilde0*np.ones(p-n+1), dtilde1))
        dtilde =dtilde.reshape((len(dtilde),1))

    if dtilde.ndim ==1:
        dtilde = dtilde.reshape((len(dtilde),1))

    dhat = pav(dtilde)
    sigmahat = u.dot(np.tile(dhat,(1,p))*(u.T))

    return sigmahat

def LW_est(X):
    '''
    LW optimal shrinkage coeff estimate
    X:(n_samples, n_features)
    '''
    lw = LedoitWolf()
    cov_lw = lw.fit(X).covariance_

    return cov_lw

def OAS_est(X):
    '''
    OAS coef estimate
    X:(n_samples, n_features)
    '''
    oa = OAS()
    cov_oa = oa.fit(X).covariance_

    return cov_oa

def Garch(X, K, method=1):

    n, p = X.shape
    
    cond_vol = np.zeros((n,p))
    cond_vol[:] = np.nan

    if method == 1:
        pre_vol = np.zeros((K, p))
        pre_vol[:] = np.nan

        for j in range(p):
            ind = np.where(~np.isnan(X[:,j]))[0]
            xx = X[ind, j]

            md = arch_model(xx - np.mean(xx), mean='Zero', p=1, q=1)
            res = md.fit(disp = 'off', show_warning=False)
            cond_vol[ind, j] = res.conditional_volatility
            pre_vol[:, j] = md.forecast(res.params, horizon=K).variance.values[-1, :].values

        cond_vol = pd.DataFrame(cond_vol).interpolate().fillna(method='bfill').values

    if method == 2:
        for j in range(p):
            ind = np.where(~np.isnan(X[:, j]))[0]
            xx = X[ind, j]
            # xnav[ind, j] = np.cumprod(1 + xx)
            md = arch_model(xx - np.mean(xx), mean='Zero', p=1, q=1)
            res = md.fit(disp='off', show_warning=False)
            cond_vol[ind, j] = res.conditional_volatility
        cond_vol = pd.DataFrame(cond_vol).interpolate().fillna(method='bfill').values  


    XX = X - np.nanmean(X,axis = 0)
    Y = interpolate_return_2(XX/cond_vol)

    rho = direct_kernel(Y)
    aa = np.diag(1/np.sqrt(np.diag(rho)))
    rho = np.dot(aa.dot(rho),aa)

    if method == 1:
        cond_cov = np.zeros((p, p, K))
        cond_cov[:] = np.nan
        for k in range(K):
            tmp = np.diag(np.sqrt(pre_vol[k, :])).dot(rho).dot(np.diag(np.sqrt(pre_vol[k, :])))
            cond_cov[:, :, k] = (tmp + tmp.T) / 2

        return np.sum(cond_cov, axis=2)




    if method == 2:
        cond_cov2 = np.zeros((p, p, K))
        cond_cov2[:] = np.nan
        for k in range(K):
            tmp = np.diag(cond_vol[-k - 1, :]).dot(rho).dot(np.diag(cond_vol[-k - 1, :]))
            cond_cov2[:, :, k] = (tmp + tmp.T) / 2

        return np.sum(cond_cov2, axis=2)
    

def portfolio_opt(risk):
    
    P = cp.Variable(risk.shape[0])
    
    constraints = [
        0 <= P, P <= 0.5,
        cp.sum_entries(P) == 1
    ]
    
    objective = cp.Minimize(cp.quad_form(P, risk))
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    return P

def cov_NW(cov_data, lambd, delay=2):

    Tn = cov_data.shape[0]
    Fn = cov_data.shape[1]

    w = np.array([lambd**n for n in range(Tn)][::-1])
    w = w / w.sum()

    factor_mean_weighted = np.average(cov_data, axis=0, weights=w)

    f_cov_raw = np.array([cov_data[:, i] - factor_mean_weighted[i] for i in range(Fn)])

    #calc cov matrix
    F_raw = np.zeros((Fn, Fn))
    for i in range(Fn):
        for j in range(Fn):
            cov_ij = np.sum(f_cov_raw[i] * f_cov_raw[j] * w)
            F_raw[i, j] = cov_ij

    cov_nw = np.zeros((Fn, Fn))
    F_NW = 21.*F_raw

    for d in range(1, delay+1):
        cov_nw_i = np.zeros((Fn, Fn))
        for i in range(Fn):
            for j in range(Fn):
                cov_ij = np.sum(f_cov_raw[i][:-d] * f_cov_raw[j][d:] * w[d:] ) / np.sum(w[d:])
                cov_nw_i[i,j] = cov_ij

        F_NW += 21.*((1-d/(delay+1.)) * (cov_nw_i + cov_nw_i.T))

def NW_adjusted(data, tau=90, length=100, n_start=100, n_forward=21, NW=1):
    data_cov = data.iloc[n_start-length:n_start,:].as_matrix()
    
    lambd = 0.5**(1./tau)
    
    # calculate Newey-West covariance
    if NW:
        F_NW = cov_NW(data_cov,lambd)
    else:
        F_NW = np.cov(data_cov.T)*21
    
    # decomp of NW covariance 
    s, U = linalg.eigh(F_NW)
    
    r = (data.iloc[n_start:n_start+n_forward,:]+1).cumprod().iloc[-1,:]-1
    R_eigen   = np.dot(U.T,r)
    Var_eigen = s
    
    if not np.allclose(F_NW,  U @ np.diag(s) @ U.T ):
        print('ERROR in eigh')
        return
    
    return data_cov, U, F_NW, R_eigen, np.sqrt(Var_eigen)

def Eigen_adjusted(F_NW,U,Std_i,length=252,N_mc=1000):
    
    for i in range(N_mc):
        
        if i%200==0:print(i)
        
        r_mc = np.array([np.random.normal(0, std_, length) for std_ in Std_i])
        r_mc = np.dot(U,r_mc)
        
        
        F_mc = np.cov(r_mc)
        s, U_mc = linalg.eigh(F_mc)
        q = (U_mc.T@F_NW)@(U_mc)
        
        
        if i==0:
            stat = np.diagonal(q)/s
        else:
            stat+= np.diagonal(q)/s
    
    stat = np.sqrt(stat/N_mc)
    
    return stat


def Gamma_fitting(gamma_k,para=2,n_start_fitting = 16):
    # para: the scaling parameter "a"
    # n_start_fitting: assign zero weight to the first "n_start_fitting" eigenfactor, here we choose 16 while UNE4 chooses 15
    y = gamma_k[n_start_fitting:]
    x = np.array(range(n_start_fitting,40))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    gamma_k_new = [p(xi) for xi in range(n_start_fitting)]+list(y)
    
    gamma_k_new = para*(np.array(gamma_k_new)-1)+1
    
    return gamma_k_new


def v_fitting(gamma_k,amp=2,n_start_fitting = 16):

    y = gamma_k[n_start_fitting:]
    x = np.array(range(n_start_fitting,40))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    gamma_k_new = [p(xi) for xi in range(n_start_fitting)]+list(y)
    
    gamma_k_new = amp*(np.array(gamma_k_new)-1)+1
    
    return gamma_k_new

def v_fitting_modified(gamma_k,amp=1.6,n_start_fitting = 15):

    y = gamma_k[n_start_fitting:]
    x = np.array(range(n_start_fitting,40))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    gamma_k_new = [p(xi) for xi in range(n_start_fitting)]+list(y)
    gamma_k_new=np.array(gamma_k_new)+0.05
    
    gamma_k_new[:33] = amp*(np.array(gamma_k_new[:33])-1.05)+1.05
    
    return np.array(gamma_k_new)

def EWMA(n=252,tau=42,norm=1):
    
    lambd = 0.5**(1./tau)
    
    w = np.array([lambd**n for n in range(252)][::-1])
    
    if norm:
        return w/w.sum()
    else:
        return w

def cal_bais_stat(data_factor,length=252,n_forward=21,tau=90,N_mc=1000,NW=0):

    Bias=[]
    stat_all=[]
    
    for i in range(length,data_factor.shape[0]-n_forward,10):
        
        print("# %1d/%1d" %(i,data_factor.shape[0]-n_forward))
        
        data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data_factor,tau=tau,length=length,n_start=i,NW=0)
        
        Bias.append(R_i/Std_i)
        
        stat = Eigen_adjusted(F_NW,U,Std_i,length=length,N_mc=N_mc)
        stat_all.append(stat)
    
    Bias = np.array(Bias)
    bias_eigen = [np.std(Bias[:,x]) for x in range(Bias.shape[1])]
    
    return bias_eigen,stat_all
