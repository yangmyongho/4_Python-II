# -*- coding: utf-8 -*-
"""
step01_regression

회귀방정식에서 기울기(slope)와 절편(intercept) 식
    기울기(slope) = Cov(x, y) / Sxx(x의 편차 제곱 평균)
    절편(intercept) = y_mu - (slope * x_mu)
"""
from scipy import stats # 회귀모델
import pandas as pd # csv file read
import numpy as np 



# 1. galton.csv


# 1-1) regression
galton = pd.read_csv("C:/ITWILL/4_Python-II/data/galton.csv")
galton.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 928 entries, 0 to 927
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   child   928 non-null    float64  -> y변수 
 1   parent  928 non-null    float64  -> x변수      '''
# x, y 변수 선택
x = galton.parent
y = galton['child']
# model 생성
model = stats.linregress(x, y)
model
''' slope=0.6462905819936423,        <기울기>
    intercept=23.941530180412748,    <절편>
    rvalue=0.4587623682928238,       <설명력> : 예측값이 정답과 일치할확률 : 46%
    pvalue=1.7325092920142867e-49,   <유의성 검정> 
    stderr=0.04113588223793335       <표준오차>                            '''
# 회귀방정식 : Y = x * a(기울기) + b(절편)
y_pred = x * model.slope + model.intercept
y_pred
# 예측치  vs  관측치(정답)
y_pred.mean() # 68.08846982758534
y.mean()      # 68.08846982758512
# 평균값이 거의 일치함
err = y - y_pred # 오류 = 정답 - 예측치


# 1-2) 기울기 계산 : Cov(x, y) / Sxx(x의 편차 제곱 평균)
xu = x.mean() # = x_mu
yu = y.mean() # = y_mu
n = len(x)
Cov_xy = sum((x-xu) * (y - yu)) / n
Cov_xy # 2.062389686756837
Sxx = np.mean((x - xu)**2)
Sxx # 3.1911182743757336
slope = Cov_xy / Sxx
slope # 0.6462905819936413  <lineregress 기울기값과 거의 유사하다.>


# 1-3) 절편 계산 : y_mu - (slope * x_mu)
intercept = yu - (slope * xu)
intercept # 23.94153018041171 <lineregress 절편값과 거의 유사하다.>
 

# 1-4) 설명력(rvalue) : rvalue=0.4587623682928238
galton.corr()
'''
           child    parent
child   1.000000  0.458762
parent  0.458762  1.000000  '''

# 수식  vs  library함수
y_pred2 = x * slope + intercept
y_pred2.mean() # 68.08846982758423
y_pred.mean() # 68.08846982758534
# 거의 일치하다.





































































