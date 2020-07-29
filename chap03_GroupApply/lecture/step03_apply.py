# -*- coding: utf-8 -*-
"""
step03_apply

1. group 객체에 외부 함수 적용
 - object.apply(func1) : 하나의 외부함수
 - object.agg([func1, func2 ... ]) : 여러개의 외부함수

2. data 정규화

"""
import pandas as pd
import numpy as np # max, min, log


# 1. group 객체에 외부 함수 적용
''' apply  vs  agg
     - 공통점 : 그룹 객체에 외부 함수를 적용
     - 차이점 : 적용할 함수의 갯수 '''

# 1-1) group객체 생성
test = pd.read_csv("C:/ITWILL/4_Python-II/data/test.csv")
test.info()
grp = test['data2'].groupby(test['key'])

# 1-2) 내장함수
grp.size() # a 3, b 3
grp.sum() # a 300, b 500
grp.max() # a 100, b 200
grp.min() # a 100, b 100

# 1-3) 사용자 정의함수 생성
def diff(grp):
    result = grp.max() - grp.min()
    return result

# 1-4) apply(func1) 사용
# 1-4-1) 내장함수
grp.apply(sum) # a 300, b 500
grp.apply(max) # a 100, b 200
grp.apply(min) # a 100, b 100
# 1-4-2) 사용자함수
grp.apply(diff) # a 0, b 100

# 1-5) agg([func1, func2 ... ]) 사용
agg_func = grp.agg(['sum', 'max', 'min', diff])
agg_func
'''
     sum  max  min  diff
key                     
a    300  100  100     0
b    500  200  100   100 '''


# 2. data 정규화
# - 다양한 특징을 갖는 변수(x)를 대상으로 일정한 범위로 조정
# - x(30) -> y
# - ex) x1 : 10~100 , x2 : 0.1~10 처럼 범위가 다양한 경우

# 2-1) 사용자 함수 : 0 ~ 1
x = [10, 20000, -100, 0] # 범위가 다양함
def normal(x):
    n = (x - np.min(x)) / (np.max(x) - np.min(x))
    return n
normal(x) # [0.00547264, 1, 0, 0.00497512] : -100 -> 0 , 20000 -> 1 범위가 0~1

# 2-2) 자연 log (밑수=e) : 음수,0 인경우 -> 결측치,무한대로 바뀜
''' 로그 vs 지수 역함수 관계
     - 로그 : 지수값 반환
     - 지수 : 로그값 반환    '''
# 2-2-1) e 
e = np.exp(1)
e # 2.718281828459045
# 2-2-2) 로그 -> 지수값(8 = 2^3)
np.log(10) # 2.302585092994046 = e^2.3025
e**2.302585092994046 # 10.000000000000002 -> 10과 가까움
# 2-2-3) 지수 -> 로그값
np.exp(2.302585092994046) # 10.000000000000002
# 2-2-4) x 를 log 취함
np.log(x) # [2.30258509, 9.90348755, nan, -inf] : -100 -> nan , 0 -> -inf


# 3. iris dataset 적용
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")    
cols = list(iris.columns)
cols # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

# 3-1) Species 제거
iris_x = iris[cols[:4]]
iris_x.shape # (150, 4)
iris_x.head()
'''
   Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
2           4.7          3.2           1.3          0.2
3           4.6          3.1           1.5          0.2
4           5.0          3.6           1.4          0.2 '''

# 3-2) x변수 정규화
iris_x_nor = iris_x.apply(normal)
iris_x_nor.head()
'''
   Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
0      0.222222     0.625000      0.067797     0.041667
1      0.166667     0.416667      0.067797     0.041667
2      0.111111     0.500000      0.050847     0.041667
3      0.083333     0.458333      0.084746     0.041667
4      0.194444     0.666667      0.067797     0.041667 '''

# 3-3) agg 사용
iris_x.agg(['var', 'mean', 'max', 'min'])
'''
      Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
var       0.685694     0.189979      3.116278     0.581006
mean      5.843333     3.057333      3.758000     1.199333
max       7.900000     4.400000      6.900000     2.500000
min       4.300000     2.000000      1.000000     0.100000 '''






















