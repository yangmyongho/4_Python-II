'''
문1) score_iq.csv 데이터셋을 이용하여 단순선형회귀모델을 생성하시오.
   <조건1> y변수 : score, x변수 : academy      
   <조건2> 회귀모델 생성과 결과확인(회귀계수, 설명력, pvalue, 표준오차) 
   <조건3> 회귀선 적용 시각화 
   
문2) irsi.csv 데이터셋을 이용하여 다중선형회귀모델을 생성하시오.
   <조건1> 칼럼명에 포함된 '.' 을 '_'로 수정
   iris = pd.read_csv('../data/iris.csv')
   iris.columns = iris.columns.str.replace('.', '_')
   <조건2> y변수 : 1번째 칼럼, x변수 : 2~4번째 칼럼    
   <조건3> 회귀계수 확인 
   <조건4> 회귀모델 세부 결과 확인  : summary()함수 이용 
'''

from scipy import stats
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pylab import plot, legend, show
score_iq  = pd.read_csv("C:/ITWILL/4_Python-II/data/score_iq.csv")
score_iq.info()

# 1
y = score_iq.score
x = score_iq.academy
# 2
model = stats.linregress(x,y)
model
''' LinregressResult(slope=4.847829398324446  <기울기>, 
                     intercept=68.23926884996192  <절편>, 
                     rvalue=0.8962646792534938  <설명력>, 
                     pvalue=4.036716755167992e-54  <pvalue>, 
                     stderr=0.1971936807753301  <표준오차>)      '''
y_pred = x * model.slope + model.intercept
# 3
plt.plot(x, y, 'bo', label='x,y scatter')
plt.plot(x, y_pred, 'r.-', label='y pred')
legend(loc='best')
plt.show()


# 1
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")
iris.info()
iris.columns = iris.columns.str.replace('.', '_')
iris.info()
# 2
Y = iris.iloc[:,0]
Y
X = iris.iloc[:,1:4]
X
# 3
corr = iris.corr()
corr['Sepal_Length']
''' 
Sepal_Length    1.000000
Sepal_Width    -0.117570
Petal_Length    0.871754
Petal_Width     0.817941 '''
formula = "Sepal_Length ~ Sepal_Width + Petal_Length + Petal_Width"
model = sm.ols(formula, data=iris).fit()
model # <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x29dd5e59308>
model.params
'''
Intercept       1.855997
Sepal_Width     0.650837
Petal_Length    0.709132
Petal_Width    -0.556483 '''

# 4
model.summary()
# Adj. R-squared:                  0.856
# F-statistic:                     295.5  (-1.96~1.96이아니므로 기각)
# Prob (F-statistic):           8.59e-62  < 0.05 이므로 기각
'''
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        1.8560      0.251      7.401      0.000       1.360       2.352
Sepal_Width      0.6508      0.067      9.765      0.000       0.519       0.783
Petal_Length     0.7091      0.057     12.502      0.000       0.597       0.821
Petal_Width     -0.5565      0.128     -4.363      0.000      -0.809      -0.304
p값이 전부 0.05보다 작으므로 귀무가설 기각 
따라서 , 세변수 모두 y값에 영향을 준다
t검정통계량이 양이면 양의 상관관계 (정비례) 음이면 음의 상관관계(반비례)
셋중 제일 영향을 미치는 것은 Petal_Length (t값이 제일큼)
'''
# 예측치  vs  관측치  비교
y_pred = model.fittedvalues
y_true = iris['Sepal_Length']

# 그래프그리기
plt.plot(y_true, 'b', label='real values')
plt.plot(y_pred, 'r', label='pred values')
plt.legend(loc='best')
plt.show()
















