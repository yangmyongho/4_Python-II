# -*- coding: utf-8 -*-
"""
step03_Descriptive
 - DataFrame의 요약통계량
 - 상관계수
"""
import pandas as pd


# 1. Descriptive
product = pd.read_csv("C:\\ITWILL\\4_Python-II\\data/product.csv")
print(product.info())
print(product)
iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data/iris.csv")
''' 만약 경로가 설정 되어있다면 바로 iris.csv 로 사용가능 
    iris = pd.read_csv(iris.csv) '''

# 1-1) 요약통계량 
summ = product.describe() # Df.describe : summary와 비슷
print(summ)

# 1-2) 행/열 통계
print(product.sum(axis=0)) # 행축 열단위 합계
print(product.sum(axis=1)) # 열축 행단위 합계
print(product.mean(axis=0)) # 행축 열단위 평균
print(product.mean(axis=1)) # 열축 행단위 평균

# 1-3) 산포도 : 분산, 표준편차
print(product.var()) # 생략하면 axis = 0
print(product.std()) # 생략하면 axis = 0 

# 1-4) 빈도수 : 집단변수
cnt_a = product['a'].value_counts()
print(cnt_a) # 3 4 2 1 5 빈도수로 내림차순
cnt2_a = product.a.value_counts()
print(cnt2_a) # 위아래같음 

# 1-5) 유일값 보기
uni_c = product['c'].unique()
print(uni_c) # [3 2 4 5 1]

# 1-6) 상관관계
print(product.corr()) # 상관계수 정방행렬 < 숫자가 클수록 상관성이 높음 >

# 1-7) iris dataset 적용
# 1-7-1) subset 생성
iris_df = iris.iloc[:,:4] # 숫자이므로 0부터3까지<숫자형식만뽑기위해서>
print(iris_df.shape) # (150, 4)
# 1-7-2) 변수 4개 요약 통계량
print(iris_df.describe())
# 1-7-3) 상관계수 행렬
print(iris_df.corr())
# 1-7-4) 빈도수
species = iris.Species
print(species.value_counts())
# 1-7-5) 유일값 보기
print(species.unique()) # array 
print(list(species.unique())) # list형





































