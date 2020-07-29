# -*- coding: utf-8 -*-
"""
step01_DF_merge
 - DataFrame 병합(merge)
 ex) DF1(id) + DF2(id)  ->  DF3
"""
import pandas as pd
from pandas import Series, DataFrame


# 1. Series merge : 1차원
s1 = Series([1, 3], index=['a', 'b'])
s2 = Series([5, 6, 7], index=['a', 'b', 'c'])
s3 = Series([11, 13], index=['a', 'c'])

# 1-1) 행 단위 결합 : rbind()
s4 = pd.concat([s1,s2,s3], axis=0) # axis=0 행단위
print(s4)
print(s4.shape) # (7,)

# 1-2) 열 단위 결합 : cbind()
s5 = pd.concat([s1,s2,s3], axis=1) # axis=1 열단위
print(s5)
print(s5.shape) # (3,3)


# 2. DataFrame 병합
# 현재경로 : C:\ITWILL\4_Python-II\data
wdbc = pd.read_csv('wdbc_data.csv')
print(wdbc.info())
''' RangeIndex: 569 entries, 0 to 568
    Data columns (total 32 columns): '''

# 2-1) DF1(16) + DF2(16)
cols = list(wdbc.columns)
print(len(cols)) # 32

# 2-1-1) DF1, DF2 생성 
DF1 = wdbc[cols[:16]] # 0~15
print(DF1.shape) # (569, 16)
DF2 = wdbc[cols[16:]] # 16~31
print(DF2.shape) # (569, 16)

# 2-1-2) DF2 에 공통칼럼 추가
id = wdbc.id
DF2['id'] = id
print(DF2.shape) # (569, 17)
print(DF2.head()) # id 추가

# 2-1-3) 병합 : 공통 칼럼 이용
DF3 = pd.merge(DF1, DF2) # 공통인 id 를 통해서 merge 가능 
print(DF3.info()) # (569, 32) 원본과같음

# 2-1-4) 결합 : 칼럼 단위 결합
DF1 = wdbc[cols[:16]] # 0~15
print(DF1.shape) # (569, 16)
DF2 = wdbc[cols[16:]] # 16~31
print(DF2.shape) # (569, 16)
DF4 = pd.concat([DF1, DF2], axis=1) # 열 단위
print(DF4.info()) # (569, 32) 원본과같음









































