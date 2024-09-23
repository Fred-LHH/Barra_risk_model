# -*- coding = utf-8 -*-
# @Time: 2024/09/23
# @Author: Haohan
# @File: Quality.py
# @Software: Vscode
'''
#### 5. 质量 Quality
·Quality(1):
    ·Leverage(2)
        ·Market Leverage(3) 市场杠杆
        ·Book Leverage(3) 账面杠杆
        ·Debt to asset ratio(3) 资产负债比
    ·Earning Variability(2)
        ·Variation in Sales(3) 营业收入波动率
        ·Variation in Earning(3) 盈利波动率
        ·Variation in cashflows(3) 现金流波动率
        ·Standard deviation of analyst Earnings-to-Price(3) 分析师预测盈市率标准差
    ·Earnings Quality(2)
        ·Accruals Balancesheet version(3) 资产负债表应计项目
        ·Accruals Cashflow version(3) 现金流量表应计项目
    ## 盈利能力Profitability(二级因子)
        资产周转率Asset turnover
        资产毛利率Gross profitability
        销售毛利率Gross Profit Margin
        总资产收益率 Return on assets
    ## Investment Quality
        总资产增长率Total Assets Growth Rate：最近5个财政年度的总资产对时间的回归的斜率值，除以平均总资产，最后取相反数
        股票发行量增长率Issuance growth：最近5个财政年度的流通股本对时间的回归的斜率值，除以平均流通股本，最后取相反数
        资本支出增长率Capital expenditure growth：将过去5个财政年度的资本支出对时间的回归的斜率值，除以平均资本支出，最后取相反数
        资本支出是指用于购买各种长期资产(长期投资固定资产无形资产和其他长期资产)的支出然后再减去无息长期负债(各种不需支付利息的长期应付款专项应付款等)的增加额。
'''
import pandas as pd
import numpy as np
import utils as ut
from tqdm import tqdm

def calc_Leverage(data: pd.DataFrame):
    data['PE'] = (data['oth_eqt_tools_p_shr'] / 1e8).fillna(0)
    data['LD'] = (data['total_ncl'] / 1e8).fillna(0)
    data['f_ann_date'] = pd.to_datetime(data['f_ann_date'])
    data['total_mv'] = data.groupby('code')['total_mv'].shift(1)
    data['total_mv'] = data['total_mv'] / 1e4 
    data['end_date'] = pd.to_datetime(data['end_date'])
    data['discDate'] = data['end_date'].apply(ut._discDate)
    data = data.query('f_ann_date<discDate').drop(columns=['ann_date', 'end_date', 'report_type', 'end_type', 'total_ncl', 'oth_eqt_tools_p_shr'])
    data = ut._pubDate_align_tradedate(data, 'discDate', '20240831')
    data.rename(columns={'total_mv': 'ME'}, inplace=True)
    data['BE'] = data['ME'] / data['pb']
    data['Market_Leverage'] = data.eval('(ME+PE+LD)/ME')
    data['Book_Leverage'] = data.eval('(BE+PE+LD)/ME')
    data['Debt_to_asset_ratio'] = data.eval('total_liab/total_assets')
    data = data[['code', 'trade_date', 'Market_Leverage', 'Book_Leverage', 'Debt_to_asset_ratio']]
    return data

def calc_Earning_Variability(data: pd.DataFrame, forecast: pd.DataFrame, mv: pd.DataFrame):
    # 用TTM数据计算
    data = data[['ts_code', 'f_ann_date_x', 'end_date', 'revenue', 'net_profit', 'n_incr_cash_cash_equ']].rename(columns={'f_ann_date_x':'ann_date', 'ts_code': 'code'})
    data.drop_duplicates(subset=['code', 'end_date'], keep='last', inplace=True)
    data['end_date'] = pd.to_datetime(data['end_date'])
    data = data[data['end_date'].dt.month.isin([6, 12])]
    basic = ut._calculate_ttm(data)
    basic['ann_date'] = pd.to_datetime(basic['ann_date'])
    basic['discDate'] = basic['end_date'].apply(ut._discDate)
    basic = basic.query('ann_date<discDate').drop(columns=['ann_date']).rename(
    columns={'discDate':'trade_date'}).sort_values(['code', 'end_date'])
    def variation(x, **kargs):
        return 4*np.nanstd(x) / np.nanmean(x)

    def _modify(x):
        vars_ = ['Variation_in_Sales', 'Variation_in_Earning', 'Variation_in_Cashflow']
        for v in vars_:
            x[v] = ut._winsorize(x[v].fillna(np.nanmedian(x[v])))
            x[v] -= x[v].mean()
            x[v] /= x[v].std()
        return x

    V_in_Sales = ut._panel_rolling_apply(basic, value_col='revenue_TTM', window=5, apply_func=variation).rename(columns={'revenue_TTM':'Variation_in_Sales'})
    V_in_Earning = ut._panel_rolling_apply(basic, value_col='net_profit_TTM', window=5, apply_func=variation).rename(columns={'net_profit_TTM':'Variation_in_Earning'})
    V_in_Cashflow = ut._panel_rolling_apply(basic, value_col='n_incr_cash_cash_equ_TTM', window=5, apply_func=variation).rename(columns={'n_incr_cash_cash_equ_TTM':'Variation_in_Cashflow'})
    factor = V_in_Sales.merge(V_in_Earning, how='outer').merge(V_in_Cashflow, how='outer')
    factor = factor.groupby('trade_date').apply(_modify).reset_index(drop=True)
    factor = ut._pubDate_align_tradedate(factor, 'trade_date', '20240831')
    
    for year in tqdm(range(2014, 2025), desc='计算Standard deviation of analyst Earnings-to-Price...'):
        mask = (forecast['Fenddt'] == pd.to_datetime('{}1231'.format(year)))
        tmp = forecast[mask].copy()
        tmp['Fnetpro'] /= 1e8
        np_std = tmp.groupby('Stkcd').apply(ut._cumstd).reset_index()
        np_std.rename(columns={'Stkcd': 'code', 'Fenddt':'end_date'}, inplace=True)
        np_std = ut._pubDate_align_tradedate(np_std, pubDate_col=None, end_date=str(year)+'1231', analysis_pred=True)
        total_mv = mv[mv['trade_date'].dt.year == year]
        np_std = np_std.merge(total_mv, on=['code', 'trade_date'])
        np_std['forecast_EP_std'] = np_std.eval('np_std/total_mv')
        forecast_EP_std.append(np_std)
    forecast_EP_std = pd.concat(forecast_EP_std, axis=0)
    forecast_EP_std = forecast_EP_std[['code', 'trade_date', 'forecast_EP_std']]
    factor = factor.merge(forecast_EP_std, on=['code', 'trade_date'], how='outer')
    return factor


def calc_Earnings_Quality(data: pd.DataFrame):
    data['discDate'] = data['end_date'].apply(ut._discDate)
    data = data.query('ann_date<discDate').drop(columns='ann_date')\
        .rename(columns={'discDate':'ann_date'})\
        .sort_values(by=['code', 'end_date'])
    data['year'] = data['end_date'].apply(lambda x: x.year)
    data.rename(columns={'daa':'DA', 'total_assets': 'TA', 'total_liab': 'TL', 'end_bal_cash_equ': 'Cash', 'n_cashflow_act':'CFO', 'n_cashflow_inv_act':'CFI', 'net_profit': 'NI'}, inplace=True)
    diff_col = ['st_borr', 'total_ncl', 'non_cur_liab_due_1y', 'CFO', 'CFI', 'NI']
    def _diff(x):
        if len(x)==2:    
            x.loc[x.index[-1], diff_col] = x.loc[:, diff_col].diff().iloc[-1, :]
        return x
    data = data.groupby(['code', 'year']).apply(_diff).reset_index(drop=True)
    data = data.sort_values(by=['code', 'end_date'])
    data['TD'] = data.fillna(0).eval('st_borr+total_ncl+non_cur_liab_due_1y')
    data['NOA'] = data.fillna(0).eval('(TA-Cash)-(TL-TD)')
    data['delta_NOA'] = data.groupby('code')['NOA'].diff()
    data['ACCR_BS'] = data.eval('delta_NOA-DA')
    data['ABS'] = data.eval('-ACCR_BS/TA')
    data['ACCR_CF'] = data.fillna(0).eval('NI-(CFO+CFI)+DA')
    data['ACF'] = data.fillna(0).eval('-ACCR_CF/TA')
    factor = data[['code', 'ann_date', 'ABS', 'ACF']]
    factor = ut._pubDate_align_tradedate(factor, 'ann_date', '20240831')
    return factor

def cal_Investment_Quality(data):
    periods = ['{}{}'.format(year, date) for year in range(2009, 2025) for date in ['0331', '0630', '0930', '1231']]
    periods = periods[:-2]
    periods = pd.to_datetime(periods)
    data['end_date'] = pd.to_datetime(data['end_date'])

    def _sub_calc_IQ(time):
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
        Total_Assets_Growth_Rate = - tmp.groupby('code').apply(
            ut._t_reg, field='total_assets', min_period=6).fillna(0.)
        Issuance_growth = - tmp.groupby('code').apply(
            ut._t_reg, field='float_share', min_period=6).fillna(0.)
        
        Capital_expenditure_growth = - tmp.groupby('code')\
            .apply(ut._t_reg, field='CTD', min_period=2).fillna(0.)
        sub_factor = pd.concat([Total_Assets_Growth_Rate, Issuance_growth, Capital_expenditure_growth], axis=1)
        sub_factor.columns = ['Total_Assets_Growth_Rate', 'Issuance_growth', 'Capital_expenditure_growth']
        return sub_factor
    
    factor = []
    for date in tqdm(periods, desc='计算投资质量...:'):
        if date.year < 2014:
            continue
        elif date.month != 6 and date.month != 12:
            continue
        else:
            sub_factor = _sub_calc_IQ(date)
            if date.month == 6:
                trade_date = str(date.year) + '0901'
                sub_factor['trade_date'] = trade_date
            elif date.month == 12:
                trade_date = str(date.year+1) + '0501'
                sub_factor['trade_date'] = trade_date
            factor.append(sub_factor.reset_index())
    factor = ut._pubDate_align_tradedate(pd.concat(factor), 'trade_date', '20240831')
    return factor.reset_index(drop=True)   



