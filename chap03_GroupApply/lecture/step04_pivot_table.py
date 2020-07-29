# -*- coding: utf-8 -*-
"""
step04_pivot_table
 - 사용자가 행,열 그리고 교차셀에 변수를 지정하여 테이블 생성

"""
import pandas as pd


# 1. pivot_data.csv
'''
교차셀(values) : 매출액(price)
행(index) : 년도(year), 분기(quarter)
열(columns) : 매출규모(size)
셀에 적용할 통계 : sum '''

# 1-1) pivot dataset 생성
pivot_data = pd.read_csv("C:/ITWILL/4_Python-II/data/pivot_data.csv")
pivot_data.info() # (8, 4)
pivot_data
'''   year quarter   size  price
0  2016      1Q  SMALL   1000
1  2016      1Q  LARGE   2000
2  2016      2Q  SMALL   1200
3  2016      2Q  LARGE   2500
4  2017      3Q  SMALL   1300
5  2017      3Q  LARGE   2200
6  2017      4Q  SMALL   2300
7  2017      4Q  LARGE   2800 '''

# 1-2) pivot table 생성
ptable = pd.pivot_table(pivot_data, values='price', index=['year', 'quarter'],
                        columns='size', aggfunc='sum')
ptable
'''
size          LARGE  SMALL
year quarter              
2016 1Q        2000   1000
     2Q        2500   1200
2017 3Q        2200   1300
     4Q        2800   2300 '''     
ptable.shape # (4, 2)
ptable.plot(kind='barh', title='2016vs2017', stacked=True)


# 2. movie_rating.csv
'''
교차셀 : 평점(rating)
행 : 평가자(critic)
열 : 영화제목(title)
셀에 적용할 통계 : sum '''

# 2-1) dataset 생성
movie = pd.read_csv("C:/ITWILL/4_Python-II/data/movie_rating.csv")
movie.info()
movie

# 2-2) pivot table
ptable2 = pd.pivot_table(movie, values='rating', index='critic', columns='title',
                         aggfunc='sum')
ptable2
'''
title    Just My  Lady  Snakes  Superman  The Night  You Me
critic                                                     
Claudia      3.0   NaN     3.5       4.0        4.5     2.5
Gene         1.5   3.0     3.5       5.0        3.0     3.5
Jack         NaN   3.0     4.0       5.0        3.0     3.5
Lisa         3.0   2.5     3.5       3.5        3.0     2.5
Mick         2.0   3.0     4.0       3.0        3.0     2.0
Toby         NaN   NaN     4.5       4.0        NaN     1.0 '''
ptable2.shape # (9, 6)
ptable2.plot(kind='barh', title='MOVIE CRITIC')











































