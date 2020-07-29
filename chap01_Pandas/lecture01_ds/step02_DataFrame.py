# -*- coding: utf-8 -*-
"""
step02_DataFrame

DataFrame 자료구조 특징 
 - 2차원 행렬구조(table 유사함)
 - 열(칼럼) 단위 데이터 처리 용이
 - Series(1차원)의 모음 -> DataFrame(2차원)
"""
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy.random import choice
help(choice) # 잘모를때
#choice(a, size=None, replace=True, p=None)


# 1. DataFrame 생성

# 1-1) 기본 자료구조(list, dict) 이용
name = ['hong', 'lee', 'kang', 'yoo'] # list
age = [35, 45, 55, 25] # list
pay = [350, 450, 550, 250]
addr = ['서울시', '부산시', '대전시', '인천시']
data = {'name':name, 'age':age, 'pay':pay, 'addr':addr} # dict
print(data)
frame = pd.DataFrame(data=data, columns=['name', 'age', 'addr', 'pay']) # 칼럼 순서변경가능
print(frame)
# - 칼럼 추출
print(age)
age2 = frame.age
print(age2)
age3 = frame['age']
print(age3)
# - 추출하는 이유
print(age.mean()) # list 오류
print(age2.mean()) # 40.0
print(type(age2)) # <class 'pandas.core.series.Series'>
# - 새 칼럼 추가
gender = Series(['남자', '남자', '남자', '여자'])
frame['gender'] = gender
print(frame)

# 1-2) numpy 이용 : 선형대수 관련 함수
frame2 = DataFrame(np.arange(12).reshape(3,4),columns=['a','b','c','d'], 
                   index=['A','B','C']) # 0~11까지 3행4열로 DataFrame 만듬
print(frame2)
'''    a  b   c   d
    A  0  1   2   3
    B  4  5   6   7
    C  8  9  10  11 '''
# - 행/열 통계 구하기
print(frame2.mean(axis=0)) # axis=0 행축 : 열단위
''' a    4.0
    b    5.0
    c    6.0
    d    7.0 '''
print(frame2.mean(axis=1)) # axis=1 열축 : 행단위
''' A    1.5
    B    5.5
    C    9.5 '''


# 2. DataFrame 칼럼 참조
print(frame2.index) # 행 이름  Index
print(frame2.values) # 값 이름  array
# emp.csv 파일 load
emp = pd.read_csv("C:\\ITWILL\\4_Python-II\\data/emp.csv", encoding='utf-8')
print(emp.info()) # str(emp) 과 같다.
print(emp)
print(emp.head()) # head(emp) 과 같다.
iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data/iris.csv")
print(iris.info())
print(iris.head())

# 2-1) 단일 칼럼 선택
print(emp['No'])
print(emp.No) # 칼럼명에 점포함된 경우 
print(emp.No[1]) # 102 <특정 행에 특정 칼럼 선택 = 특정원소선택>
print(iris['Sepal.Width'])
#print(iris.Sepal.Width) <오류 : 칼럼이름에 . 이있는경우 사용불가능>
# - 그래프그리기
no = emp.No
no.plot()
pay = emp['Pay']
pay.plot()

# 2-2) 복수 칼럼 선택 : 중첩 list
print(emp[['No','Pay']])
#print(emp[['No':'Name']]) # 연속기호 오류 
print(emp[['No','Name']])
# - 그래프그리기
emp[['No','Pay']].plot() # 범례까지 나옴
emp.plot() # Name 은 그래프로 나오지 않음

# 3. subset 만들기 : old DF -> new DF

# 3-1) 특정 칼럼 제외 : 칼럼 적은 경우
subset1 = emp[['Name', 'Pay']]
print(subset1)

# 3-2) columns 이용 : 칼럼 많은 경우
cols = list(iris.columns) # 칼럼명 추출
print(cols) # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
print(type(cols)) # <class 'list'>  list구조이기에 : - 사용가능
iris_x = cols[:4]
iris_y = cols[-1]
print(iris_x) # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
print(iris_y) # Species
X = iris[iris_x]
Y = iris[iris_y]
print(X)
print(Y)
print(X.shape) # (150, 4) : 2차원
print(Y.shape) # (150,) : 1차원

# 3-3) 특정 행 제외
subset2 = emp.drop(2)
print(subset2) # 강감찬행 지워짐

# 3-4) 조건식으로 행 선택 boolean
subset3 = emp[emp.Pay > 350]
print(subset3)


# 4. DataFrame 행렬 참조 : DF[row(행), col(열)]
emp
''' No Name  Pay
    0  101  홍길동  150
    1  102  이순신  450
    2  103  강감찬  500
    3  104  유관순  350
    4  105  김유신  400
    열 이름 : No Name  Pay
    행 이름 : 0 ~ 4  '''

# 1) DF.loc[row, col] : label index
print(emp.loc[1:3]) # loc가 붙으면 이름으로 설정 (1,2,3)행
print(emp.loc[1:3, :]) # 위아래 같다 < : 전부 >
print(emp.loc[1:3, 'No':'Name']) # <연속> 이름에도 : 사용 가능 
print(emp.loc[1:3, :'Name']) # 위아래 같다 < : 앞에 생략가능 >
#print(emp.loc[1:3, 'No','Name']) # 오류 < , 때문 >
print(emp.loc[1:3, ['No','Name']]) # <불연속>  , 사용하려면 [] 로 묶어준다
#print(emp.loc[1:3, [1:2]]) # 오류 <숫자형식사용불가능>

# 2) DF.iloc[row, col] : integer index
print(emp.iloc[1:3]) # iloc가 붙으면 숫자로 설정 (1,2)행
print(emp.iloc[1:3, :]) # 위아래 같다 < : 전부 >
print(emp.iloc[1:3, 0:2]) # <연속> 숫자로여겨서 0~1까지 No,Name 만 선택
print(emp.iloc[1:3, :2]) # 위아래 같다. < : 앞에 생략가능 >
#print(emp.iloc[1:3, 0,2]) # 오류 < , 때문 >
print(emp.iloc[1:3, [0,2]]) # <불연속>  , 사용하려면 [] 로 묶어준다
#print(emp.iloc[1:3, ['No','Name']]) # 오류 <이름형식사용불가능>


# 5. DF 행렬 참조 example
print(iris.shape) # (150, 5)

# 5-1) train dataset
row_idx = choice(a=len(iris), 
                 size=int(len(iris)*0.7),
                 replace=False)
print(row_idx)
print(len(row_idx)) # 105 <150*0.7>
# 5-1-1) loc 사용 (이름으로 인식)
train_set = iris.loc[row_idx]
print(train_set)
print(train_set.shape) # (105, 5)
# 5-1-2) iloc 사용 (숫자로 인식)
train_set2 = iris.iloc[row_idx]
print(train_set2)
print(train_set.shape2) # (105, 5)

# 5-2) test dataset : list + for
row2_idx = [i for i in range(len(iris)) if not i in row_idx] # row_idx 가아닌 
print(len(row2_idx)) # 45
# 5-2-1) loc 사용 (이름으로 인식)
test_set = iris.loc[row2_idx]
print(test_set)
print(test_set.shape) # (45, 5)
# 5-2-1) iloc 사용 (숫자로 인식)
test_set2 = iris.iloc[row2_idx]
print(test_set2)
print(test_set2.shape) # (45, 5)

# 5-3) x,y 변수 분리 <loc 만 가능> 
cols = list(iris.columns)
x = cols[:4]
y = cols[-1]
iris_x = iris.loc[row2_idx, x]
iris_y = iris.loc[row2_idx, y]
print(iris_x.shape) # (45, 4)
print(iris_y.shape) # (45, )


ㄱㄴㄷㄺㄴㄷㄻㅄㅇㅈㅍㅊㅋㅌㅊㅍㅀ









