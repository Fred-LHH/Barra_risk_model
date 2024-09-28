# -*- coding = utf-8 -*-
# @Time: 2024/09/26
# @Author: Haohan
# @File: Value.py
# @Software: Vscode
'''
## 估值 Value

账面市值比（Book to price）：将最近报告期的普通股账面价值除以当前市值
Earnings-to-price Ratio：过去12个月的盈利除以当前市值
分析师预测EP比：预测12个月的盈利除以当前市值
Cash earnings to price：过去12个月的现金盈利除以当前市值
Enterprise multiple：上一财政年度的息税前利润（EBIT）除以当前企业价值（EV）
长期相对强度
（1）计算非滞后的长期相对强度：对股票对数收益率进行加权求和，时间窗口1040个交易日，半衰期260个交易日

（2）滞后273个交易日，在11个交易日的时间窗口内取非滞后值等权平均值，最后取相反数

长期历史Alpha
（1）计算非滞后的长期历史Alpha：取CAPM回归（见BETA）的截距项，时间窗口1040个交易日，半衰期260个交易日

（2）滞后273个交易日，在11个交易日的时间窗口内取非滞后值等权平均值，最后取相反数

为了保证因子覆盖度，这里长期指标均采用750为回望窗口大小。
'''
import pandas as pd
import numpy as np
import utils as ut
from tqdm import tqdm

def calc_Valuefactor(data):
    # input data columns : code	ann_date end_date n_cashflow_act_TTM trade_date	pe_ttm pb total_mv
    data['total_mv'] = data['total_mv'] * 1e4
    data['Book_to_price'] = 1 / data['pb']
    data['Earning_to_price'] = 1 / data['pe_ttm']
    data['Cash_earning_to_price'] = data['n_cashflow_act_TTM'] / data['total_mv']
    return data

def calc_forecast_EP_mean(forecast, indicator):
    # indicator columns: code trade_date total_mv
    forecast_EP_mean = []
    for year in tqdm(range(2014, 2025), desc='分析师预测EP比...'):
        mask = (forecast['Fenddt'] == pd.to_datetime('{}1231'.format(year)))
        tmp = forecast[mask].copy()
        tmp['Fnetpro'] /= 1e8
        tmp.rename(columns={'Stkcd': 'code', 'Fenddt':'end_date'}, inplace=True)
        np_mean = tmp.groupby('code').apply(ut._cummean).reset_index()
        np_mean = ut._pubDate_align_tradedate(np_mean, pubDate_col=None, end_date=str(year)+'1231', analysis_pred=True)
        total_mv = indicator[indicator['trade_date'].dt.year == year]
        np_mean = np_mean.merge(total_mv, on=['code', 'trade_date'])
        np_mean['forecast_EP_mean'] = np_mean.eval('np_mean/total_mv')
        forecast_EP_mean.append(np_mean)
    forecast_EP_mean = pd.concat(forecast_EP_mean, axis=0)
    forecast_EP_mean = forecast_EP_mean[['code', 'trade_date', 'forecast_EP_mean']]
    return forecast_EP_mean

def calc_enterprise_multiple(data, indicators):
    data.rename(columns={'ts_code':'code', 'f_ann_date':'ann_date'}, inplace=True)
    data['ann_date'] = pd.to_datetime(data['ann_date'])
    data['end_date'] = pd.to_datetime(data['end_date'])
    data['discDate'] = data['end_date'].apply(ut._discDate)
    data.rename(columns={'ts_code':'code'}, inplace=True)
    data = data.query('ann_date<discDate').drop(columns='ann_date')\
        .rename(columns={'discDate':'ann_date'})\
        .sort_values(by=['code', 'end_date'])
    data = ut._pubDate_align_tradedate(data, 'ann_date', '20240831')
    data['code'] = data['code'].apply(lambda x: x.split('.')[0])
    data = pd.merge(data, indicators, on=['code', 'trade_date'])
    data['ebit'] = data['ebit'] / 1e8
    data['end_bal_cash_equ'] = data['end_bal_cash_equ'] / 1e8
    data['total_liab'] = data['total_liab'] / 1e8
    data['EV'] = data.eval('total_liab + total_mv- end_bal_cash_equ')
    data['Enterprise_multiple'] = data.eval('ebit / EV')
    Enterprise_multiple = data[['code', 'trade_date', 'Enterprise_multiple']]
    return Enterprise_multiple

def _calc_LP_Relative_strength(ret, window=750, half_life=260):
    W = ut._get_exp_weight(window=window, half_life=half_life)
    relative_strength = {}
    for i in tqdm(range(len(ret) - window - 1), desc='长期非滞后相对强度……'):
        tmp = ret.iloc[i:i+window, :]
        tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / window < 0.1].fillna(0.)
        relative_strength[tmp.index[-1]] = pd.Series(np.sum(W.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns)
    relative_strength = pd.DataFrame(relative_strength).T
    relative_strength.index.name = 'date'
    relative_strength = relative_strength.shift(273)
    relative_strength = relative_strength.rolling(11).mean().dropna(how='all').mul(-1)
    relative_strength = pd.melt(relative_strength.reset_index(), id_vars='date', value_name='LP_Relative_strength').dropna().reset_index(drop=True)
    relative_strength.columns = ['date', 'code', 'LP_Relative_strength']
    return relative_strength

def _calc_Long_Alpha(close_code, close_index, window=750, half_life=260):
    price = pd.concat([close_code, close_index], axis=0).reset_index(drop=True)
    price['ret'] = price['close'] / price['pre_close'] - 1
    ret = pd.pivot_table(price, values='ret', index='trade_date', columns='ts_code')
    W = ut._get_exp_weight(window=window, half_life=half_life)

    def _calc_Alpha(tmp):
        W_f = np.diag(W)
        Y_f = tmp.dropna(axis=1).drop(columns='399300.SZ')
        idx_f, Y_f = Y_f.columns, Y_f.values
        X_f = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
        beta_f = np.linalg.pinv(X_f.T @ W_f @ X_f) @ X_f.T @ W_f @ Y_f
        alpha_f = pd.Series(beta_f[0], index=idx_f, name=tmp.index[-1])
        
        alpha_l = {}
        for c in set(tmp.columns) - set(idx_f) - set('399300.SZ'):
            tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
            tmp_.loc[:, 'W'] = W
            tmp_ = tmp_.dropna()
            W_l = np.diag(tmp_['W'])
            if len(tmp_) < half_life:
                continue
            X_l = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
            Y_l = tmp_[c].values
            beta_l = np.linalg.pinv(X_l.T @ W_l @ X_l)@ X_l.T @ W_l @ Y_l
            alpha_l[c] = beta_l[0]
        alpha_l = pd.Series(alpha_l, name=tmp.index[-1])  
        alpha = pd.concat([alpha_f, alpha_l]).sort_index()
        return alpha
    
    Alpha = Parallel(6, verbose=10)(delayed(_calc_Alpha)(
        ret.iloc[i:i+window, :].copy()) for i in 
        tqdm(range(len(ret)-window+1), desc='正在计算alpha...'))

    Alpha = pd.concat(Alpha, axis=1).T
    Alpha = Alpha.apply(pd.to_numeric, errors='coerce')
    Alpha = Alpha.shift(273)
    Alpha = Alpha.rolling(11).mean().dropna(how='all', axis=1).mul(-1)
    Alpha = pd.melt(Alpha.reset_index(), id_vars='index', value_name='Longterm_Alpha').dropna().reset_index(drop=True)
    Alpha.columns = ['date', 'code', 'Longterm_Alpha']
    return Alpha
    



