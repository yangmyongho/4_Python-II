# -*- coding: utf-8 -*-
"""
step02_reshape
 - DataFrame 모양 변경
"""
import pandas as pd


# 1. buy dataset 모양 변경 
buy = pd.read_csv('buy_data.csv')
print(buy.info())
print(buy.shape) # (22, 3)
print(type(buy)) # <class 'pandas.core.frame.DataFrame'>

# 1-1) row -> column(wide -> long)
buy_long = buy.stack()
print(buy_long.shape) # (66, )
print(buy_long) 
''' 0   Date           20150101
        Customer_ID           1
        Buy                   3
    1   Date           20150101
        Customer_ID           2
        Buy                   4
                    ...
    20  Date           20150101 
        Customer_ID           1
        Buy                   9
    21  Date           20150107
        Customer_ID           5
        Buy                   7
Length: 66, dtype: int64 '''

# 1-2) column -> row(long -> wide)
buy_wide = buy_long.unstack()
print(buy_wide.shape) # (22, 3)
print(buy_wide) # 원래대로 돌아옴

# 1-3) 전치행렬 : t() -> .T
wide_t = buy_wide.T
print(wide_t.shape) # (3, 22)
print(wide_t) # 행 -> 열 , 열 -> 행

# 1-4) 중복 행 제거
print(buy.duplicated()) # 중복 확인 (10, 16)
buy_df = buy.drop_duplicates()
print(buy_df.shape) # (20, 3)
print(buy_df) # 중복된거 (10,16)2개 지워짐
























