'''
문) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성 
  조건1> train/test - 7:3비울
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> 모델 평가 : MSE, r2_score
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. data load
boston = load_boston()
print(boston)


# 2. 변수 선택  
x = boston.data
y = boston.target
x.shape # (506, 13)
y.shape # (506,)


# 3. train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
x_train.shape # (354, 13)
x_test.shape # (152, 13)
y_train.shape # (354,)
y_test.shape # (152,)



# 4. 회귀모델 생성 : train set
obj = LinearRegression()
model = obj.fit(x_train, y_train)
model # copy_X=True, fit_intercept=True, n_jobs=None, normalize=False
y_pred = model.predict(x_test)
y_pred.shape # (152,)


# 5. 모델 평가 : test set
mse = mean_squared_error(y_test, y_pred)
mse # 28.40585481050824
score = r2_score(y_test, y_pred)
score # 0.6485645742370703  <68%>


# 6. y_true  vs  y_pres  시각화
fig = plt.figure(figsize = (10, 5))
chart = fig.subplots()
chart.plot(y_test, color='b', label='real values')
chart.plot(y_pred, color='r', label='fitted values')
plt.legend(loc='best')
plt.title('real values  vs  fitted values')
plt.xlabel('index')
plt.ylabel('prediction')
plt.show()











