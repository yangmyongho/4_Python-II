# -*- coding: utf-8 -*-
"""
step01_Series.py

Series 객체 특징
 - 1차원의 배열구조
 - 수학/통계 함수 제공
 - 범위 수정, 블럭 연산
 - indexing/slicing 기능
 - 시계열 데이터 생성
"""
import pandas as pd # 별칭
from pandas import Series # from 패키지 import 클래스


# 1. Series 객체 생성

# 1-1) list 이용
lst = [4000, 3000, 3500, 2000]
ser = pd.Series(lst) # list -> Series 
print('lst =', lst)
print('ser =\n', ser)
print(ser.index) # 색인
# RangeIndex(start=0, stop=4, step=1)
print(ser.values) # 값  [4000 3000 3500 2000]
print(ser[0]) # 4000
ser1_2 = Series([4000, 3000, 3500, 2000], index=['a','b','c','d'])
print(ser1_2.index) # Index(['a', 'b', 'c', 'd'], dtype='object')
print(ser1_2.values) # [4000 3000 3500 2000]

# 1-2) dict 이용
person = {'name':'홍길동', 'age':35, 'addr':'서울시'}
ser2 = Series(person)
print('person =', person)
print('ser2 =\n', ser2)
print(ser2.index) # 색인 
# Index(['name', 'age', 'addr'], dtype='object')
print(ser2.values) # 값  ['홍길동' 35 '서울시']
# - index 사용 : object[n or 조건식]
print(ser2['age']) # 35
# - boolean 조건식
print(ser[ser >= 3500]) # 0 4000   2 3500  


# 2. indexing : list 동일 < 마이너스인덱스사용불가 >
ser3 = Series([4, 4.5, 6, 8, 10.5]) # 생성자
print(ser3[0]) # 4.0
print(ser3[:3]) # 0 ~ 2
print(ser3[3:]) # 3 ~ 4
print(ser3[-2]) # 오류 마이너스 인식못함 
print(ser3[:-2]) # 0 ~ 2
print(ser3[-2:]) # 3 ~ 4


# 3. Series 결합과 NA 처리
p1 = Series([400, None, 350, 200], index=['a', 'b', 'c', 'd'])
p2 = Series([400, 150, 350, 200], index=['a', 'c', 'd', 'e'])
# - Series 결합
p3 = p1+p2
print(p3)
'''
a    800.0  <400 + 400>
b      NaN  <None + 350>
c    500.0  <350 + 150>
d    550.0  <200 + 350>
e      NaN  <None + 200>
dtype: float64
'''


# 4. 결측치 처리방법 (평균, 0, 제거)
print(type(p3)) # <class 'pandas.core.series.Series'>

# 4-1) 평균 대체
p4 = p3.fillna(p3.mean()) # 616.666667
print(p4)
p4_2 = p3.fillna(p3.var()) # 분산
print(p4_2) # 25833.333333

# 4-2) 0 대체
p5 = p3.fillna(0) # 0
print(p5)

# 4-3) 결측치 제거
p6 = p3[pd.notnull(p3)] # subset
print(p6) # a,c,d


# 5. 범위수정, 블럭 연산
print(p2)

# 5-1) 범위수정 
p2[1:3] = 300
print(p2)
# - list 에서는 범위수정 불가능 
lst = [1,2,3,4]
#lst[1:3] = 3 <오류>
print(lst) # [1,2,3,4] 

# 5-2) 블럭연산
print(p2 + p2) # 2배
print(p2 - p2) # 0

# 5-3) broadcast 연산(1차 vs 0차)
v1 = Series([1,2,3,4])
scala = 0.5
b = v1 * scala # vector(1)*scala(0)
print(b)
for i in v1:
    c = i * scala
    print(c)

# 5-4) 수학/통계 함수 
print('sum =', v1.sum()) # sum = 10
print('mean =', v1.mean()) # mean = 2.5
print('var =', v1.var()) # var = 1.6666666666666667
print('std =', v1.std()) # std = 1.2909944487358056
print('max =', v1.max()) # max = 4
# 호출 가능한 멤버 확인
print(dir(v1))
print(v1.shape) # (4,)
print(v1.size) # 4




















