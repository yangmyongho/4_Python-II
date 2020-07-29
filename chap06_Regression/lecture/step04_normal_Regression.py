# -*- coding: utf-8 -*-
"""
step04_normal_Regression

data scaling(정규화, 표준화) : 이물질 제거
 - 용도 : 특정 변수의 값에 따라서 model에 영향을 미치는 경우
     ex) 범죄율(-0.1 ~ 0.99), 주택가격(99 ~ 999)
 - 정규화 : <x변수> 변수의 값을 일정한 범위로 조정(0 ~ 1, -1 ~ +1)
 - 표준화 : <y변수> 평균(=0)과 표준편차(=1)를 이용
     표준화 공식 z = (x - mu) / sd
"""
from sklearn.datasets import load_boston # 실습용 dataset
from sklearn.model_selection import train_test_split # split
from sklearn.linear_model import LinearRegression # model 생성
from sklearn.metrics import mean_squared_error,r2_score # model 평가
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 1. 


# 1-1) dataset load
'''
boston = load_boston()
x = boston.data
y = boston.target
x.shape # (506, 13)
y.shape # (506,)
'''
X, y = load_boston(return_X_y=True)
X.shape # (506, 13)
y.shape # (506,)


# 1-2) data scaling
''' X : 정규화(0 ~ 1)
    Y : 표준화(평균=0, 표준편차=1)   '''
X.max() # 711.0
X.mean() # 70.07396704469443
y.max() # 50.0
y.mean() # 22.532806324110677
# 1-2-1) 정규화함수
def normal(x):
    nor = (x - np.min(x)) / (np.max(x)- np.min(x))
    return nor
# x변수 정규화
x_nor = normal(X)
x_nor.mean() # 0.09855691567467571

# 1-2-2) 표준화함수
def zscore(y): 
    mu = y.mean()
    z = (y - mu) / y.std()
    return z
# y변수 표준화(mu=0, std=1)
y_nor = zscore(y)
y_nor.mean() # -5.195668225913776e-16 -> 0에 수렴
y_nor.std() # 0.9999999999999999 -> 1에 수렴


# 1-3) dataset split
x_train, x_test, y_train, y_test = train_test_split(x_nor, y_nor, 
                                                    random_state=123) # size생략시0.25
x_train.shape # (379, 13)
x_test.shape # (127, 13)


# 1-4) model 생성
obj = LinearRegression()
model = obj.fit(X=x_train, y=y_train)
model # copy_X=True, fit_intercept=True, n_jobs=None, normalize=False
y_pred = model.predict(X=x_test)


# 1-5) model 평가
mse = mean_squared_error(y_test, y_pred)
mse # 0.2933980240643525   <오차율 : 30% 정도>
r2sc = r2_score(y_test, y_pred)
r2sc # 0.6862448857295749  <정확율 : 68% 정도>













































































