'''
#### 3. 流动性 Liquidity
Liquidity(1):
    Liquidity(2)
        ·Monthly share turnover(3) 月换手率.对最近21个交易日的股票换手率求和,然后取对数
        ·Quarterly share turnover(3) 季换手率 T=3
        ·Annual share turnover(3) 年换手率 T=12
        ·Annualized traded value ratio(3) 年化交易量比率. 对日换手率进行加权求和,时间窗口为252个交易日,半衰期为63个交易日
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils as ut

def Liquidity(data: pd.DataFrame):
    '''
    data: 日换手率数据
    columns - code  index - trade_date  values - turnover_rate
    PS: 换手率为百分比形式，我们需要对其进行转换
    '''
    tmp = pd.pivot_table(data, index='trade_date', columns='code', values='turnover_rate')
    # 月、季、年换手率
    monthly_share_turnover = np.log(tmp.rolling(21).sum())
    idx = list(range(20, 252, 21))
    quarterly_share_turnover, annual_share_turnover = {}, {}

    for i in tqdm(range(len(tmp) - 251), desc='计算季度、年度换手率...'):
        t = tmp.index[i+251]
        mst = np.exp(monthly_share_turnover.iloc[i:i+252, :].iloc[idx, :])
        quarterly_share_turnover[t] = np.log(mst.iloc[-3:, :].mean(axis=0))
        annual_share_turnover[t] = np.log(mst.mean(axis=0))
    quarterly_share_turnover = pd.DataFrame(quarterly_share_turnover).T
    annual_share_turnover = pd.DataFrame(annual_share_turnover).T
    quarterly_share_turnover.index.name = 'trade_date'
    annual_share_turnover.index.name = 'trade_date'

    monthly_share_turnover = pd.melt(monthly_share_turnover.reset_index(), id_vars='trade_date', value_name='Monthly_share_turnover').dropna()
    quarterly_share_turnover = pd.melt(quarterly_share_turnover.reset_index(), id_vars='trade_date', value_name='Quarterly_share_turnover').dropna()
    annual_share_turnover = pd.melt(annual_share_turnover.reset_index(), id_vars='trade_date', value_name='Annual_share_turnover').dropna()

    factor = monthly_share_turnover.merge(quarterly_share_turnover).merge(annual_share_turnover)

    # 年化交易量比率
    W = ut._get_exp_weight(252, 63)
    annualized_traded_value_ratio = []
    for i in tqdm(range(len(tmp)-251), desc='计算年化交易量比率...'):
        tmp_ = tmp.iloc[i:i+252, :].copy()
        annualized_traded_value_ratio.append(
            pd.Series(np.nansum(tmp_.values * W.reshape(-1, 1), axis=0), index=tmp.columns, name=tmp_.index[-1])
        )

    annualized_traded_value_ratio = pd.concat(annualized_traded_value_ratio, axis=1).T
    annualized_traded_value_ratio.index.name = 'trade_date'
    annualized_traded_value_ratio = pd.melt(annualized_traded_value_ratio.reset_index(), id_vars='trade_date', value_name='Annualized_traded_value_ratio').dropna()
    factor = factor.merge(annualized_traded_value_ratio)
    return factor