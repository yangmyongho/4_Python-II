# -*- coding: utf-8 -*-
"""
step02_dot_regression

행렬곱 함수(np.dot) 이용 -> y 예측치 구하기
    y_pred = np.dot(X, a) + b

np.dot(X, a) 전제조건 
    1. X, a : 행렬 구조
    2. 수일치 : X열 차수 = a행 차수
"""
import numpy as np
from scipy import stats # 단순회귀모델
from statsmodels.formula.api import ols # 다중회귀모델
import pandas as pd



# 1.score_iq.csv
score = pd.read_csv("C:/ITWILL/4_Python-II/data/score_iq.csv")
score.info()
'''
 1   score    150 non-null    int64  -> y
 2   iq       150 non-null    int64  -> x1
 3   academy  150 non-null    int64  -> x2           '''
formula = "score ~ iq + academy"
model = ols(formula, data=score).fit() # 다중회귀모형
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x14a58e7ea88>

# 회귀계수 : 기울기, 절편
model.params
''' Intercept    25.229141
    iq            0.376966
    academy       2.992800   '''
# model 결과 확인 
model.summary() # t검정통계량이 클수록 영향이 크다
# R-squared:                       0.946
'''
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     25.2291      2.187     11.537      0.000      20.907      29.551
iq             0.3770      0.019     19.786      0.000       0.339       0.415
academy        2.9928      0.140     21.444      0.000       2.717       3.269
==============================================================================
Omnibus:                       36.342   Durbin-Watson:                   1.913
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               54.697
Skew:                           1.286   Prob(JB):                     1.33e-12
Kurtosis:                       4.461   Cond. No.                     2.18e+03
'''
# model 예측치 
model.fittedvalues
'''
0      83.989997
1      75.342705
2      73.457874
3      82.105166
4      64.810583  '''


# 행렬곱 : y_pred = np.dot(X, a) + b 
#         y_pred = (X1 * a1 + X2 * a2) + b   <위아래식같다>
# < 여러개의 입력값(x값) 이 있는 다중일경우 행렬곱을 쓰면 간단해진다.>
# X : iq = x1  ,  academy = x2
X = score[['iq', 'academy']]
X.shape # (150, 2) 
# list -> numpy : (2, 1)로 만들어야된다. <조건 : 수일치>
a = np.array([[0.376966], [2.992800]]) 
a.shape # (2, 1)
matmul = np.dot(X, a) # 행렬곱
matmul.shape # (150, 1) 
b = 25.229141 # 절편
# 예측치
y_pred = matmul + b # broadcast(2차원 + 0차원)
y_pred.shape # (150, 1)
# 차원변경 : 2차원 -> 1차원
y_pred1 = y_pred.reshape(150)
y_pred1.shape # (150,)
y_pred1
# 관측치
y_true = score.score
# DataFrame(관측치, 예측치)
df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred1})
df.head()
'''
   y_true     y_pred
0      90  83.989981
1      75  75.342691
2      77  73.457861
3      83  82.105151
4      65  64.810571   '''
# 오차
err = y_true - y_pred1
err
err.mean() # 1.3959999998955178e-05

# cor로 확인
cor = df['y_true'].corr(df['y_pred'])
cor # 0.9727792069594754 : 상관계수 <1에가까울수록 정답과 거의 일치하다>








































