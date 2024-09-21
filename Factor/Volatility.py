'''
#### 2. 波动率 Volatility
·Volatility(1):
    Beta(2)
        Beta(3) 股票收益率对沪深300收益率进行时间序列回归,取回归系数,回归时间窗口为252个交易日,半衰期63个交易日
    ·Residual Volatility(2)
        ·Hist sigma(3) 再计算Beta所进行的时序回归中,取回归残差收益率的波动率
        ·Daily std(3) 日收益率在过去252个交易日的波动率, 半衰期42个交易日
        ·Cumulative range(3) 累计收益范围(按月收益计算)
PS:
股票收益率数据存在缺失值，这将导致回归参数估计失败（矩阵运算结果为NaN）。所以，在任意交易日，我们筛选出具有完整数据的股票，进行批量的加权最小二乘估计；而不具有完整数据的股票，则分别取出，将数据对齐，再进行参数估计。如果最近252个交易日内，股票样本点少于63个，则不进行估计。
'''

import pandas as pd
import numpy as np
import utils as ut
from tqdm import tqdm

def calc_Beta_HS(close_code: pd.DataFrame, close_index: pd.DataFrame, window: int = 252, half_life: int = 63):
    '''
    计算Beta和Hist_sigma
    close_code: 股票close, pre_close
    close_index: 指数close, pre_close
    '''
    price = pd.concat([close_code, close_index], axis=0).reset_index(drop=True)
    price['ret'] = price['close'] / price['pre_close']
    ret = pd.pivot_table(price, values='ret', index='trade_date', columns='code')
    W = ut._get_exp_weight(window, half_life)

    def _calc_factor(tmp):
        # 不存在缺失值的股票
        W = np.diag(W)
        Y_f = tmp.dropna(axis=1).drop(columns='399300.SZ')
        idx_f, Y_f = Y_f.columns, Y_f.values
        X_f = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
        beta_f = np.linalg.pinv(X_f.T @ W @ X_f) @ X_f.T @ W @ Y_f
        hist_sigma_f = pd.Series(np.std(Y_f - X_f @ beta_f, axis=0), index=idx_f, name=tmp.index[-1])
        beta_f = pd.Series(beta_f[1], index=idx_f, name=tmp.index[-1])
        # 存在缺失值的股票
        






def Volatility(close_code: pd.DataFrame, close_index: pd.DataFrame, window: int = 252, half_life: int = 63):
    '''
    close_code: 股票close, pre_close
    close_index: 指数close, pre_close
    '''
    # 计算BETA和Hist_sigma
    beta, hist_sigma = [], []
    for i in tqdm(range(len(ret) - window + 1), desc='开始计算beta...'):
        tmp = ret.iloc[i:i+window, :].copy()
        Weights = np.diag(W)
        Y_f = tmp.dropna(axis=1).drop(columns='399300.SZ')
        idx_f, Y_f = Y_f.columns, Y_f.values
        X_f = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
        beta_f = np.linalg.pinv(X_f.T @ Weights @ X_f) @ X_f.T @ Weights @ Y_f
        hist_sigma_f = pd.Series(np.std(Y_f - X_f @ beta_f, axis=0), index=idx_f, name=tmp.index[-1])
        beta_f = pd.Series(beta_f[1], index=idx_f, name=tmp.index[-1])

        # 对于不具有完整数据的股票, 则取出进行数据对齐, 如果最近252个交易日内股票样本点少于63, 则不进行估计
        beta_l, hist_sigma_l = {}, {}
        for c in set(tmp.columns) - set(idx_f) - set('399300.SZ'):
            tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
            tmp_.loc[:, 'Weight'] = W
            tmp_.dropna(inplace=True)
            W_l = np.diag(tmp_['Weight'])
            if len(tmp_) < half_life:
                continue
            X_l = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
            Y_l = tmp_[c].values
            beta_tmp = np.linalg.pinv(X_l.T @ W_l @ X_l) @ X_l.T @ W_l @ Y_l
            hist_sigma_l[c] = np.std(Y_l - X_l @ beta_tmp)
            beta_l[c] = beta_tmp[1]
        beta_l = pd.Series(beta_l, name=tmp.index[-1])
        hist_sigma_l = pd.Series(hist_sigma_l, name=tmp.index[-1])
        beta.append(pd.concat([beta_f, beta_l]).sort_index())
        hist_sigma.append(pd.concat([hist_sigma_f, hist_sigma_l]).sort_index())
    beta = pd.concat(beta, axis=1).T
    beta = pd.melt(beta.reset_index(), id_vars='index').dropna()
    beta.columns = ['trade_date', 'code', 'BETA']
    hist_sigma = pd.concat(hist_sigma, axis=1).T
    hist_sigma = pd.melt(hist_sigma.reset_index(), id_vars='index').dropna()
    hist_sigma.columns = ['trade_date', 'code', 'Hist_sigma']
    factor = pd.merge(beta, hist_sigma) # BETA和Hist_sigma
    #### 计算Daily std
    ### 采用EWMA估计股票ret的波动率,半衰期42个交易日
    init_std = ret.std(axis=0)
    L = 0.5 ** (1 / 42)
    init_var = ret.var(axis=0)
    tmp = init_var.copy()
    daily_std = {}
    for t, k in tqdm(ret.iterrows(), desc='计算Daily std...'):
        tmp = tmp * L + k ** 2 * (1 - L) #EWMA
        daily_std[t] = np.sqrt(tmp)
        tmp = tmp.fillna(init_var)
    daily_std = pd.DataFrame(daily_std).T
    daily_std.index.name = 'trade_date'
    daily_std = pd.melt(daily_std.reset_index(), id_vars='trade_date', value_name='Daily_std').dropna()
    daily_std.columns = ['trade_date', 'code', 'Daily_std']
    factor = factor.merge(daily_std)
    #### 计算Cumulative range
    close = pd.pivot_table(price, values='close', index='trade_date', columns='code').fillna(method='ffill', limit=10)
    pre_close = pd.pivot_table(price, values='pre_close', index='trade_date', columns='code').fillna(method='ffill', limit=10)
    idx = close.index
    CMRA = {}
    for i in tqdm(range(252, len(close)), desc='计算CMRA...'):
        close_ = close.iloc[i-window:i, :]
        pre_close_ = pre_close.iloc[i-window, :]
        pre_close_.name = pre_close_.name - pd.Timedelta(days=1)
        close_ = pd.concat([close_, pre_close_.to_frame().T], axis=0).sort_index().iloc[list(range(0, 253, 21)), :]
        r_tau = close_.pct_change().dropna(how='all')
        Z_T = np.log(r_tau + 1).iloc[::-1].cumsum(axis=0)
        CMRA[idx[i-1]] = Z_T.max(axis=0) - Z_T.min(axis=0)

    CMRA = pd.DataFrame(CMRA).T
    CMRA.index.name = 'trade_date'
    CMRA = pd.melt(CMRA.reset_index(), id_vars='trade_date', value_name='Cumulative_range').dropna()
    factor = factor.merge(CMRA)
    return factor






