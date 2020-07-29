# -*- coding: utf-8 -*-
"""
문01) 이항검정 : 토요일(Sat)에 오는 여자 손님 중 비흡연자가 흡연자 보다 많다고 할 수 있는가?

 # 귀무가설 : 비흡연자와 흡연자의 비율은 차이가 없다.(P=0.5)
"""

import pandas as pd
from scipy import stats
import numpy as np

tips = pd.read_csv("../data/tips.csv") # workspace 일때
print(tips.info())
print(tips.head())

day = tips['day']
day # 244개 day 값
print(day.value_counts())
'''
Sat     87  -> 토요일 빈도수 
Sun     76
Thur    62
Fri     19
'''

gender = tips['sex']
print(gender.value_counts())
'''
Male      157
Female     87 -> 여자 빈도수
'''

# 풀이 
Fe_sat = tips.loc[np.logical_and(tips['day']=='Sat', tips['sex']=='Female')]
Fe_sat
len(Fe_sat) # 28
smoker = Fe_sat['smoker']
print(smoker.value_counts()) # Yes : 15  ,  No : 13(성공횟수)

pvalue = stats.binom_test(x=13, n=28, p=0.5, alternative='two-sided')
pvalue # 0.8505540192127226
# pvalue > 0.05 따라서 귀무가설채택수
# 비흡연자와 흡연자의 비율은 차이가 없다.
pvalue2 = stats.binom_test(x=15, n=28, p=0.5, alternative='two-sided')
pvalue2 # 0.8505540192127226


pvalue == pvalue2 # True












