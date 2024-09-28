'''
Grwoth

分析师预测长期盈利增长率:分析师预测的长期(3-5)年利润增长率
每股收益增长率:过去5个财政年度的每股收益对时间回归的斜率除以平均每股年收益
每股营业收入增长率:过去5个财政年度的每股年营业收入对时间回归斜率除以平均每股年营业收入
'''

import pandas as pd
import numpy as np
import utils as ut
from tqdm import tqdm

def calc_forecast_profit_growth(forecast):
    forecast_roe_mean = []
    for year in tqdm(range(2014, 2025)):
        years = ['{}1231'.format(year+i) for i in range(3)]
        mask = (forecast['Fenddt'].isin(years))
        tmp = forecast[mask].copy()
        tmp['base_date'] = pd.to_datetime(str(year)+'1231')
        growth_mean = tmp.groupby('Stkcd').apply(ut._cummean, 'FROE', multi_periods=True).reset_index()
        growth_mean.rename(columns={'Stkcd': 'code', 'base_date':'end_date'}, inplace=True)
        growth_mean = growth_mean[['code', 'end_date', 'np_mean']]
        growth_mean = ut._pubDate_align_tradedate(growth_mean, pubDate_col=None, end_date=str(year)+'1231', analysis_pred=True)
        forecast_roe_mean.append(growth_mean)
    forecast_roe_mean = pd.concat(forecast_roe_mean)
    forecast_roe_mean.sort_values(['code', 'trade_date'], inplace=True)
    forecast_roe_mean.reset_index(drop=True, inplace=True)
    return forecast_roe_mean

def calc_growth_factor(data):
    periods = ['{}{}'.format(year, date) for year in range(2009, 2025) for date in ['0331', '0630', '0930', '1231']]
    periods = periods[:-2]
    periods = pd.to_datetime(periods)
    data['end_date'] = pd.to_datetime(data['end_date'])
    data = ut._calculate_ttm(data)

    def _sub_calc_factor(time):
        # 这里的date为半年度频率
        # 获取date的前19个季度的数据
        idx = periods.get_loc(time)
        dates = periods[idx-19:idx+1]
        df = data.loc[data['end_date'].isin(dates)].copy()
        df['discDate'] = df['end_date'].apply(ut._discDate)
        df = df.query('ann_date<discDate').drop(columns='ann_date')\
            .rename(columns={'discDate':'trade_date'})\
            .sort_values(by=['code', 'end_date'])
        tmp = df.copy()
        tmp.reset_index(drop=True, inplace=True)
        revenue_Growth_Rate = - tmp.groupby('code').apply(
            ut._t_reg, field='revenue_TTM', min_period=6).fillna(0.)
        income_growth = - tmp.groupby('code').apply(
            ut._t_reg, field='n_income_attr_p_TTM', min_period=6).fillna(0.)
        
        sub_factor = pd.concat([revenue_Growth_Rate, income_growth], axis=1)
        sub_factor.columns = ['revenue_Growth_Rate', 'income_growth']
        return sub_factor
    
    factor = []
    for date in tqdm(periods, desc='计算Growth...:'):
        if date.year < 2014:
            continue
        elif date.month != 6 and date.month != 12:
            continue
        else:
            sub_factor = _sub_calc_factor(date)
            if date.month == 6:
                trade_date = str(date.year) + '0901'
                sub_factor['trade_date'] = trade_date
            elif date.month == 12:
                trade_date = str(date.year+1) + '0501'
                sub_factor['trade_date'] = trade_date
            factor.append(sub_factor.reset_index())
    factor = ut._pubDate_align_tradedate(pd.concat(factor), 'trade_date', '20240831')
    return factor.reset_index(drop=True)   










