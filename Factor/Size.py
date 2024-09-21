'''
#### 1. 市值因子 Size
·Size(1):
    ·Size(2)
        ·LNCAP(3) 流通市值的自然对数
    ·Mid Cap(2)
        ·MIDCAP(3) 中市值.首先取Size exposure的立方, 再WLS对Size正交, 最后去极值, 标准化
'''

import pandas as pd
import numpy as np
import utils as ut


def Size(data: pd.DataFrame):
    '''
    传入带有流通市值的DataFrame
    columns: code date circ_mv
    '''
    data['circ_mv'] = data['circ_mv'] / 1e4 # 以亿元为单位
    data['LNCAP'] = np.log(data['circ_mv']+1)
    data['sub_MIDCAP'] = data['LNCAP'] ** 3
    # 截面正交
    MIDCAP = data.groupby('trade_date').apply(ut._regress, 'sub_MIDCAP', 'LNCAP', 'circ_mv')
    MIDCAP.name = 'MIDCAP'
    data = data.merge(MIDCAP.reset_index())
    data['Size'] = (data['LNCAP'] + data['MIDCAP']) / 2
    return data[['code', 'trade_date', 'LNCAP', 'MIDCAP', 'Size']]