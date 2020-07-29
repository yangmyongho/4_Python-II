# -*- coding: utf-8 -*-
"""
step01_Matplot

 - matplotlib API 사용 차트 그리기
     형식) plt.plot(data) ; plt.show()
1. 기본차트 그리기 : 선차트 : plt.plot() , 막대차트 : plt.bar()
2. 산점도 그리기 : plt.scatter()
3. subplot 이용한 차트 그리기 : 격자형식 차트
"""
import matplotlib.pyplot as plt # 별칭: plt
# data -> 한글차트.txt 에서가져옴
# 차트에서 한글 지원 
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np # 숫자 데이터 생성
help(plt.plot)
''' plot(x, y)        # plot x and y using default line style and color
    plot(x, y, 'bo')  # plot x and y using blue circle markers
    plot(y)           # plot y using x as index array 0..N-1
    plot(y, 'r+')     # ditto, but with red plusses '''
dir(plt)  # plt 에서 호출할수있는 함수들
''' 그림종류 : bar, barh, boxplot, hist, pie, scatter, 
    설정종류 : legend, label, lim, scale, ticks '''

    
# 1. 기본차트 그리기
data = np.arange(10) # 0 ~ 9
data2 = np.random.randn(10) # 평균 = 0, 표준편차 = 1 : 표준정규분포

# 1-1) plot(x, y)
plt.plot(data, data2) # (x,y)
plt.show()

# 1-2) plot(x, y, 'bo')
plt.plot(data, data2, 'bo') # b : blue , o : point모양 o
plt.show()

# 1-3) plot(y)
plt.plot(data)
plt.show()

# 1-4) plot(y, 'r+')
plt.plot(data, 'r+')
plt.show()


# 2. 산점도 그리기
data3 = np.random.randint(1, 4, 10) # 1 ~ 3 임의의 난수 10개
data3 # array([3, 3, 1, 2, 3, 1, 3, 1, 1, 1])

# 2-1) 단색 산점도  <1-2)와 비슷>
plt.scatter(data, data2, c='b', marker='o')
plt.show()

# 2-2) 여러가지 색상 산점도 <그룹별(군집별) 색상>
plt.scatter(data, data2, c=data3, marker='o')
plt.show()


# 3. subplot 이용 차트 그리기
data4 = np.random.randint(1,100, 100) # 1 ~ 99 임의의 난수 100개
data5 = np.random.randint(10,110, 100) # 10 ~ 109 임의의 난수 100개
data6 = np.random.randint(1,4, 100) # 1 ~ 3 임의의 난수 100개

# 3-1) 격자생성 (차트크기)
fig = plt.figure(figsize=(5,3)) # plt.figure(figure=(가로,세로)) 사이즈 지정
x1 = fig.add_subplot(2,2,1) # (행,열,셀)
x2 = fig.add_subplot(2,2,2)
x3 = fig.add_subplot(2,2,3)
x4 = fig.add_subplot(2,2,4)

# 3-2) 첫번째 격자 : 히스토그램 
x1.hist(data4)
#plt.show()

# 3-3) 두번째 격자 : 산점도
x2.scatter(data4, data5, c=data6)
#plt.show()

# 3-4) 세번째 격자 : 선 그래프
x3.plot(data4)
#plt.show()

# 3-5) 네번째 격자 : 점선그래프
x4.plot(data5, 'g--')
plt.show()


# 4. 차트 크기 지정, 두개 이상 차트 그리기
data = np.arange(10) # 0 ~ 9
data2 = np.random.randn(10)

# 격자생성 (차트크기)
fig2 = plt.figure(figsize=(12,5))
chart = fig2.add_subplot() # (1,1,1)

# 4-2) 계단형 차트
chart.plot(data, color='r', label='step', drawstyle='steps-post')

# 4-3) 선스타일 차트
chart.plot(data2, color='b', label='line')

# 4-4) 차트 제목,이름
plt.title('계단형  vs  선스타일') # 제목
plt.xlabel('데이터') # x축 이름
plt.ylabel('난수 정수') # y축 이름
plt.legend(loc='best') # 범례위치 : best -> 가장 적절한 자리에 
plt.show()


# 5. 막대차트
data = [127, 90, 202, 150, 250]
idx = range(len(data)) # 0 ~ 4

# 5-1) 격자생성 (차트크기생략)
fig3 = plt.figure() # 사이즈 생략하면 기본사이즈
chart2 = fig3.add_subplot() # (1,1,1)

# 5-2) 막대그래프 생성
chart2.bar(idx, data, color='darkblue')
#plt.show()

# 5-3) x축 눈금 레이블
x_label = ['서울', '대전', '부산', '광주', '인천']
plt.xticks(idx, x_label)
plt.xlabel('판매 지역')
plt.ylabel('지역별 매출현황')
plt.title('2020년도 1분기 전국 지역별 판매현황')
plt.show()






