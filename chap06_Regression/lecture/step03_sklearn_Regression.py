# -*- coding: utf-8 -*-
"""
step03_sklearn_Regression

sklearn 관련 Linear Regression
    sklearn 특징 : 정형화 되어있다.
"""
from sklearn.linear_model import LinearRegression # model object
from sklearn.model_selection import train_test_split # train/test split
from sklearn.metrics import mean_squared_error,r2_score # model 평가
from sklearn.datasets import load_diabetes # 당뇨병 데이터셋 가져오기
import numpy as np # 숫자 처리
import pandas as pd # 상관계수
import matplotlib.pyplot as plt # 회귀선시각화



# 1. 당뇨병(diabetes) 단순선형회귀모델 : x(1) -> y


# 1-1) dataset load
X, y = load_diabetes(return_X_y=True)
X.shape # (442, 10)
y.shape # (442,)
y.mean() # 152.13348416289594


# 1-2) x,y 변수
# x(bmi : 비만도지수)  ->  y
x_bmi = X[:,2]
x_bmi.shape #(442,)
x_bmi = x_bmi.reshape(442, 1) # 1차원 -> 2차원
x_bmi.shape #(442, 1)


# 1-3) model 생성 : object -> traing -> model
obj = LinearRegression() # 생성자 -> object
model = obj.fit(x_bmi, y) # (X, y) -> model   < X는 2차원요구>
model # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
# 1-3-1) y 예측치
y_pred = model.predict(x_bmi) # predict(X)
y_pred.shape # (442,)

# 1-4) model 평가 : MSE(y변수 = 정규화 인경우), r2_score(y변수 = 비정규화 인경우)
MSE = mean_squared_error(y, y_pred) # (y정답, y예측치)
print('MSE =', MSE) # MSE = 3890.4565854612724
score = r2_score(y, y_pred) # (y정답, y예측치)
print('r2_score =',score) # r2_score = 0.3439237602253803


# 1-5) dataset split(70 : 30)
x_train, x_test, y_train, y_test = train_test_split(x_bmi, y, test_size=0.3) # (X, y, test비율)
x_train.shape # (309, 1)
x_test.shape # (133, 1)
y_train.shape # (309,)
y_test.shape # (133,)


# 1-6) model 생성
obj = LinearRegression() # 생성자 -> object
model2 = obj.fit(x_train, y_train) # traing dataset
# 1-6-1) y 예측치
y_pred2 = model2.predict(x_test) # test dataset
y_pred2.shape # (133,)


# 1-7) model 평가 : MSE(정규화), r2_score(비정규화)
MSE2 = mean_squared_error(y_test, y_pred2)
print('MSE =', MSE2) # MSE = 3910.4951808769574 (오차: 3910)
score2 = r2_score(y_test, y_pred2)
print('r2_score =',score2) # r2_score = 0.27725182324415276 (예측력 : 27%)
y_test[:10]
y_pred2[:10]


# 1-8) DataFrame 
df = pd.DataFrame({'y_true':y_test, 'y_pred':y_pred2})
df
cor = df['y_true'].corr(df['y_pred']) # 상관계수
cor # 0.5502254179608806  <정확도 : 55%>


# 1-9) 산점도,회귀선 시각화
plt.plot(x_test, y_test, 'ro') # 산점도
plt.plot(x_test, y_pred2, 'b-') # 회귀선
plt.show()



# 2. iris 다중선형회귀모델 : x(3) -> y(1)


# 2-1) dataset load
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")
iris.info()
'''
 0   Sepal.Length  150 non-null    float64   -> y
 1   Sepal.Width   150 non-null    float64   -> x1
 2   Petal.Length  150 non-null    float64   -> x2
 3   Petal.Width   150 non-null    float64   -> x3
 4   Species       150 non-null    object
'''


# 2-2) x,y 변수 선택
cols = list(iris.columns)
cols # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
y_cols = cols[0] # 'Sepal.Length'
x_cols = cols[1:-1] # ['Sepal.Width', 'Petal.Length', 'Petal.Width']


# 2-3) datset split (70 : 30)
iris_train, iris_test = train_test_split(iris, test_size=0.3, random_state=123) 
'''
 test_size : 검정데이터셋 비율 (생략시 0.25로 고정)
 random_state : sampling seed값 (입력해주면 값이 항상 고정) 
'''
iris_train.shape # (105, 5)
iris_test.shape # (45, 5)
iris_train.head() 
'''
     Sepal.Length  Sepal.Width  Petal.Length  Petal.Width     Species
114           5.8          2.8           5.1          2.4   virginica
136           6.3          3.4           5.6          2.4   virginica
53            5.5          2.3           4.0          1.3  versicolor
19            5.1          3.8           1.5          0.3      setosa
38            4.4          3.0           1.3          0.2      setosa
'''
iris_test.head()
'''
     Sepal.Length  Sepal.Width  Petal.Length  Petal.Width     Species
72            6.3          2.5           4.9          1.5  versicolor
112           6.8          3.0           5.5          2.1   virginica
132           6.4          2.8           5.6          2.2   virginica
88            5.6          3.0           4.1          1.3  versicolor
37            4.9          3.6           1.4          0.1      setosa
'''


# 2-4) model 생성 : train data
lr = LinearRegression() # 생성자 -> object (이름 바꿔도됨)
model = lr.fit(iris_train[x_cols], iris_train[y_cols])
model # copy_X=True, fit_intercept=True, n_jobs=None, normalize=False
y_pred = model.predict(iris_test[x_cols]) # 예측치
y_pred.shape # (45,)
y_true = iris_test[y_cols] # 관측치(정답)
y_true.shape # (45,)
y_true.min() # 4.3
y_true.max() # 7.9


# 2-5) model 평가 : test data
MSE = mean_squared_error(y_true, y_pred)
print('MSE =', MSE) # MSE = 0.11633863200224723 (0기준 오차: 0.12)
score = r2_score(y_true, y_pred)
print('r2_score =',score) # r2_score = 0.8546807657451759 (1기준 예측력: 85%)
'''
MSE = 평균제곱오차 : mean((y_true - y_pred)**2)
r2_score = 결정계수 : 1 기준 
'''


# 2-6) pandas -> numpy < 시각화하기위해 >
type(y_true) # pandas.core.series.Series
y_true = np.array(y_true)
type(y_true) # numpy.ndarray


# 2-7) y_true  vs  y_pred  시각화
fig = plt.figure(figsize = (10, 5))
chart = fig.subplots()
chart.plot(y_true, color='b', label='real values')
chart.plot(y_pred, color='r', label='fitted values')
plt.legend(loc='best')
plt.show()










