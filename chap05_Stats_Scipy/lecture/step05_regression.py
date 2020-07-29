# -*- coding: utf-8 -*-
"""
step05_regression

scipy 패키지의 stats모듈의 함수
 - 통계적인 방식의 회귀분석
1. 단순선형회귀모델
2. 다중선형회귀모델
"""
from scipy import stats # 회귀모델 생성
import pandas as pd # csv file read
from pylab import plot, legend, show
from statsmodels.formula.api import ols # function
import matplotlib.pyplot as plt



# 1. 단순선형회귀모델

# 1-1) score_iq.csv
''' x -> y  '''
score_iq  = pd.read_csv("C:/ITWILL/4_Python-II/data/score_iq.csv")
score_iq.info()
# 변수 선택
x = score_iq.iq
y = score_iq['score']
# 회귀모델 생성 
model = stats.linregress(x, y)
model
'''
LinregressResult(slope=0.6514309527270075, <기울기>
                 intercept=-2.8564471221974657, <절편>
                 rvalue=0.8822203446134699, <설명력>
                 pvalue=2.8476895206683644e-50, <x의 유의성 검정>
                 stderr=0.028577934409305443 <표본 오차> ) '''
print('x 기울기 :', model.slope) # x 기울기 : 0.6514309527270075
print('y 절편 :', model.intercept) # y 절편 : -2.8564471221974657
score_iq.head(1)
'''      sid  score   iq  academy  game  tv
    0  10001     90  140        2     1   0  '''
# y = X * a + b
X = 140
y_pred = X * model.slope + model.intercept
y_pred # 88.34388625958358 < 점수 예측치 > 
Y = 90
err = Y - y_pred # 오차 = 정답 - 예측치
err # 1.6561137404164157  < 오차 >


# 1-2) product.csv
product  = pd.read_csv("C:/ITWILL/4_Python-II/data/product.csv")
product.info()
product.corr() # b 와 c 가 상관계수가 높다
# 변수 선택 : x = 제품적절성  vs  y = 제품만족도
model2 = stats.linregress(product['b'], product['c'])
model2
'''
LinregressResult(slope=0.7392761785971821,
                 intercept=0.7788583344701907, 
                 rvalue=0.766852699640837, 
                 pvalue=2.235344857549548e-52, 
                 stderr=0.03822605528717565)  '''
product.head(1)
'''    a  b  c
    0  3  4  3    '''
X = 4
y_pred = X * model2.slope + model2.intercept
y_pred # 3.735963048858919
Y = 3
err2 = Y - y_pred
err2 # -0.7359630488589191
# 왜 제곱하는지 모르겠음..
X = product['b']
y_pred = X * model2.slope + model2.intercept
Y = product['c']
len(y_pred)  # 264
y_pred[:10]
Y[:10]
Y - y_pred # 오차



# 2. 회귀모델 시각화
plot(product['b'], product['c'], 'b.') # plot(x, y) - 산점도
plot(product['b'], y_pred, 'r.-')
legend(['x,y scatter', 'regress model line'])
plt.show()



# 3. 다중선형회귀모델 : y ~ x1 + x2 +...
wine = pd.read_csv("C:/ITWILL/4_Python-II/data/winequality-both.csv")
wine.info()
wine.columns = wine.columns.str.replace(' ', '_') # 컬럼명 공백 변경
wine.info()
# quality  vs  others
cor = wine.corr()
cor['quality'].sort_values(ascending=False)
'''
alcohol                 0.444319
citric_acid             0.085532
free_sulfur_dioxide     0.055463
sulphates               0.038485
              ...
chlorides              -0.200666
volatile_acidity       -0.265699    '''
formula = "quality ~ alcohol + chlorides + volatile_acidity"
model = ols(formula, data=wine).fit()
model # <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1b5ae153588>
model.summary() # 모델 분석 결과 확인
# R-squared:                       0.260 <설명력>
# F-statistic:                     758.6 <F-통계량>
# Prob (F-statistic):               0.00 <유의성 검정>
# x의 유의성 검정 
'''
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept            2.9099      0.091     31.977      0.000       2.732       3.088
alcohol              0.3196      0.008     39.415      0.000       0.304       0.335
chlorides            0.1593      0.298      0.535      0.593      -0.425       0.743
volatile_acidity    -1.3349      0.061    -21.780      0.000      -1.455      -1.215
'''
model.params # 기울기,절편
'''
Intercept           2.909941 <절편>
alcohol             0.319578 <기울기>
chlorides           0.159258 <기울기>
volatile_acidity   -1.334944 <기울기>
'''

# y의 예측치 vs y의 관측치
y_pred = model.fittedvalues # y의 예측치 
y_true = wine['quality'] # y의 관측치
# 오차 (벡터 연산가능)
err = y_true - y_pred 
err
# 평균제곱오차
err2 = (y_true - y_pred)**2
err2 

y_true.mean() # 5.818377712790519
y_pred.mean() # 5.81837771279059


# 차트 확인
plt.plot(y_true[:50], 'b', label='real values')
plt.plot(y_pred[:50], 'r', label='pred values')
plt.yticks(range(0, 10))
plt.legend(loc='best')
plt.show()











