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





