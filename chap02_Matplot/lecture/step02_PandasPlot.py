# -*- coding: utf-8 -*-
"""
step02_PandasPlot

pandas 객체에서 지원하는 시각화
 형식) object.plot(kind)   <kind=차트유형>
    object : Series or DataFrame
    kind : bar, barh, pie, hist, kde, box, scatter
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import plotting
from mpl_toolkits.mplot3d import Axes3D


# 1. Series 객체 시각화
ser = pd.Series(np.random.randn(10), index=np.arange(0,100,10)) # (시작,끝,증가값)
print(ser)

# 1-1) 기본차트 : 선그래프,파란색
ser.plot() # kind 생략 -> 기본차트
ser.plot(color='r') # 선색깔 빨간색


# 2. DataFrame 객체 시각화
df = pd.DataFrame(np.random.randn(10, 4), columns=['one', 'two', 'three', 'four'])
print(df)
df.shape # (10, 4)

# 2-1) 기본차트 : 선그래프, 색상,범례까지 자동
df.plot() # 

# 2-2-1) 세로막대차트
df.plot(kind='bar', title='세로막대차트')
# 2-2-2) 세로막대차트 누적형
df.plot(kind='bar', title='세로막대차트 누적형', stacked=True)

# 2-3-1) 가로막대차트
df.plot(kind='barh', title='가로막대차트')
# 2-3-2) 가로막대차트 누적형
df.plot(kind='barh', title='가로막대차트 누적형', stacked=True)

# 2-4) 도수분포(히스토그램)
df.plot(kind='hist', title='히스토그램')

# 2-5) 커널밀도추정 : kde -> kernel density estimate
df.plot(kind='kde', title='KDE')


# 3. tips.csv 에 적용해보기
tips = pd.read_csv("C:/ITWILL/4_Python-II/data/tips.csv")
tips

# 3-1) 요일(day)  vs  파티규모(size)  범주확인
tips['day'].unique() # array(['Sun', 'Sat', 'Thur', 'Fri'], dtype=object)
tips['size'].unique() # array([2, 3, 4, 1, 6, 5], dtype=int64)

# 3-2) 교차분할표 생성 <2개의 집단변수로 만든다>
tab = pd.crosstab(tips['day'], tips['size'])
print(tab)
type(tab) # pandas.core.frame.DataFrame
tab_result = tab.loc[:,2:5]
# tab_resilt = tab.loc[:,[2,3,4,5]] 위랑 똑같다
tab_result

# 3-3) 가로막대차트 누적형
tab_result.plot(kind='barh', title='요일(day)  vs  파티규모(size)', stacked=True)


# 4. 산점도 matrix
#from pandas import plotting
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")

# 4-1) 숫자칼럼만 추출
cols = list(iris.columns)
iris_x = iris[cols[:4]]

# 4-2) 산점도 matrix 생성
plotting.scatter_matrix(iris_x)


# 5. 3d 산점도
#from mpl_toolkits.mplot3d import Axes3D
col1 = iris[cols[0]] # 첫번째 칼럼
col2 = iris[cols[1]]
col3 = iris[cols[2]]
col4 = iris[cols[3]]

# 5-1) 색상 넣기위해 집단화하기
cdata = []
for s in iris.Species:
    if s == 'setosa':
        cdata.append(1)
    elif s == 'versicolor':
        cdata.append(2)
    else:
        cdata.append(3)

# 5-2) 격자생성
fig = plt.figure()
chart = fig.add_subplot(1,1,1, projection='3d')

# 3d 산점도 생성
chart.scatter(col1, col2, col3, c=cdata) # (x,y,z, color)
chart.set_xlabel('col1')
chart.set_ylabel('col2')
chart.set_zlabel('col3')

















