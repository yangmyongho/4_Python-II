# -*- coding: utf-8 -*-
"""
step02_groupby_plot

    집단변수 기준 자료 분석
     - subset 생성
     - group 객체 생성
     - group 객체 시각화
"""
import pandas as pd
import matplotlib.pyplot as plt


# 1. wine dataset load

# 1-1) DataFrame 생성
wine = pd.read_csv("C:/ITWILL/4_Python-II/data/winequality-both.csv")
wine.info() # DataFrame (6497, 13)

# 1-2) 컬럼명 수정
wine.columns = wine.columns.str.replace(' ','_') # ('변경전','변경후')
wine.info() # 공백 -> _ 변경

# 1-3) 집단변수 확인 (type, quality)
wine['type'].unique() # array(['red', 'white'], dtype=object)
wine.type.unique() # array(['red', 'white'], dtype=object) 
# <type 은 명령어로 오해할수있으므로 되도록이면 위에꺼 사용
wine['quality'].unique() # array([5, 6, 7, 4, 8, 3, 9], dtype=int64)
wine.quality.unique() # array([5, 6, 7, 4, 8, 3, 9], dtype=int64)


# 2. subset 생성

# 2-1) type 칼럼 : DataFrame(2차원)
red_wine = wine.loc[wine['type']=='red'] # [row, col] 인데 col 전부일경우 생략가능
red_wine.info() # DataFrame (1599, 13)
red_wine.shape # (1599, 13) 2차원

# 2-2) type(행) vs quality(열) : series(1차원)
# 2-2-1) red
red_quality = wine.loc[wine['type']=='red', 'quality']
type(red_quality) # pandas.core.series.Series
red_quality.shape # (1599, ) 1차원
# 2-2-2) white
white_quality = wine.loc[wine['type']=='white', 'quality']
type(white_quality) # pandas.core.series.Series
white_quality.shape # (4898, ) 1차원


# 3. group 객체 생성 : 집단변수 2개 -> 11개변수 그룹화
# 형식) DF.groupby(['칼럼1','칼럼2'])

# 3-1) 집단변수2개 그룹화
wine_grp = wine.groupby([wine['type'], wine['quality']])
wine_grp = wine.groupby(['wine', 'quality']) # 이렇게 생략가능 (위아래같음)

# 3-2) 그룹별 함수
# 3-2-1) 그룹별 빈도수 확인
wine_grp.size()
'''
type   quality
red    3            10
       4            53
       5           681
       6           638
       7           199
       8            18
white  3            20
       4           163
       5          1457
       6          2198
       7           880
       8           175
       9             5 '''
# 3-2-2) 교차분할표 : 1d -> 2d <가독성이 높아진다.>
grp_2d = wine_grp.size().unstack()       
grp_2d
'''
quality     3      4       5       6      7      8    9
type                                                   
red      10.0   53.0   681.0   638.0  199.0   18.0  NaN
white    20.0  163.0  1457.0  2198.0  880.0  175.0  5.0 '''
# 3-2-3) 교차분할표 : 정수형 생성(NaN 대신 0 표시)
table = pd.crosstab(wine['type'], wine['quality']) # (index=행, columns=열)
table
'''
quality   3    4     5     6    7    8  9
type                                     
red      10   53   681   638  199   18  0
white    20  163  1457  2198  880  175  5 '''


# 4. group 객체 시각화
type(grp_2d) # pandas.core.frame.DataFrame

# 4-1) 가로막대차트
grp_2d.plot(kind='barh', title='TYPE vs QUALITY', stacked=True)
# (그래프종류, 제목, 누적형유무)
plt.show()


# 5. 집단변수와 연속형변수 사용 

# 5-1) type(집단변수) vs alcohol(연속형변수)  통계량
wine_grp = wine.groupby('type') # 집단변수 1개 -> 12개 변수 그룹화
wine_grp['alcohol'].describe() # 그룹별 알코올 요약통계량
'''
        count       mean       std  min  25%   50%   75%   max
type                                                          
red    1599.0  10.422983  1.065668  8.4  9.5  10.2  11.1  14.9
white  4898.0  10.514267  1.230621  8.0  9.5  10.4  11.4  14.2 '''

















