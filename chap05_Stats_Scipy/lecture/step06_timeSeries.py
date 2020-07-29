# -*- coding: utf-8 -*-
"""
step06_timeSeries

시계열 분석(time series analysis)
1. 시계열 자료 생성
2. 날짜형식 수정 (다국어)
3. 시계열 데이터 시각화
4. 이동평균 기능 : 5, 10, 20일 평균 -> 추세선 평활(smooothing)
"""
from datetime import datetime # 날짜형식 수정
import pandas as pd # csv file read
import matplotlib.pyplot as plt # 시계열 시각화
import numpy as np # 수치 자료 생성



# 1. 시계열 자료 생성
# 일단위 시계열 자료 생성 (생략시)
time_data = pd.date_range("2017-03-01", "2020-03-30")
time_data # dtype='datetime64[ns]', length=1126, freq='D' <일단위>
# 월 단위 시계열 자료 생성 <마지막달 빼고 2월까지만>
time_data2 = pd.date_range("2017-03-01", "2020-03-30", freq='M')
time_data2 # dtype='datetime64[ns]', freq='M' <월단위>
len(time_data2) # 36
# 월 단위 매출현황(자료가 없으므로 랜덤으로 생성)
x= pd.Series(np.random.uniform(10, 100, size=36)) 
df = pd.DataFrame({'date' : time_data2, 'price' : x})
df
# 시각화 
plt.plot(df['date'], df['price'], 'g--') # plt.plot(x,y)
plt.show()



# 2. 날짜형식 수정 (다국어)
cospi = pd.read_csv("C:/ITWILL/4_Python-II/data/cospi.csv")
cospi.info()
cospi.head(3)
'''
        Date     Open     High      Low    Close  Volume
0  26-Feb-16  1180000  1187000  1172000  1172000  176906
1  25-Feb-16  1172000  1187000  1172000  1179000  128321
2  24-Feb-16  1178000  1179000  1161000  1172000  140407
   날짜 형식이  일-월-년 으로 되어있음 -> 년-월-일 로 변경       '''
date = cospi['Date']
len(date) # 247
date
# list + for : 26-Feb-16  ->  2016-02-16
kdate = [datetime.strptime(d, '%d-%b-%y') for d in date]
kdate
# 날짜 칼럼 수정
cospi['Date'] = kdate
cospi.head(3)
'''
        Date     Open     High      Low    Close  Volume
0 2016-02-26  1180000  1187000  1172000  1172000  176906
1 2016-02-25  1172000  1187000  1172000  1179000  128321
2 2016-02-24  1178000  1179000  1161000  1172000  140407    '''



# 3. 시계열 데이터 시각화
cospi.index # RangeIndex(start=0, stop=247, step=1)
# 칼럼 -> index 적용
new_cospi = cospi.set_index('Date') # 숫자 index에서 date를 index로 변경
new_cospi.index # DatetimeIndex( dtype='datetime64[ns]', name='Date', length=247, freq=None)
# 날짜로 검색
new_cospi['2016'] # 2016년 정도만 검색 가능
len(new_cospi['2016']) # 37
new_cospi['2015-03'] # 2015년 03월만 검색
len(new_cospi['2015-03']) # 22
new_cospi['2015-05':'2015-03'] # 2015년 03월~2015년05월까지 범위 검색
len(new_cospi['2015-05':'2015-03']) # 62
# subset(High, Low )
new_cospi_HL = new_cospi[['High', 'Low']]
new_cospi_HL
new_cospi_HL.index # DatetimeIndex( dtype='datetime64[ns]', name='Date', length=247, freq=None)
new_cospi_HL.columns # Index(['High', 'Low'], dtype='object')
# 2015년 기준 High,Low 시각화
new_cospi_HL['2015'].plot(title='2015 year <High vs Low>')
plt.show()
# 2016년 기준 High,Low 시각화
new_cospi_HL['2016'].plot(title='2016 year <High vs Low>')
plt.show()



# 4. 이동평균 기능 : 5, 10, 20일 평균 -> 추세선 평활

# 4-1) 5일 단위 이동평균 : 5일 단위 평균 -> 마지막 5일째 이동
roll_mean5 = pd.Series.rolling(new_cospi_HL.High, window=5, center=False).mean()
roll_mean5

# 4-2) 10일 단위 이동평균 : 10일 단위 평균 -> 마지막 10일째 이동
roll_mean10 = pd.Series.rolling(new_cospi_HL.High, window=10, center=False).mean()
roll_mean10
roll_mean10.head(15)

# 4-3) 20일 단위 이동평균 : 20일 단위 평균 -> 마지막 20일째 이동
roll_mean20 = pd.Series.rolling(new_cospi_HL.High, window=20, center=False).mean()
roll_mean20

# 4-4) rolling mean 시각화
new_cospi_HL.High.plot(color='r', label='High') # 원본
roll_mean5.plot(color='orange', label='rolling mean 5day') # 5일단위=1주
roll_mean10.plot(color='g', label='rolling mean 10day') # 10일단위=2주
roll_mean20.plot(color='b', label='rolling mean 20day') # 20일단위=4주=1달
plt.legend(loc='best')
plt.show()















































