'''   
문5) iris.csv 파일을 읽어와서 다음과 같이 처리하시오.
   조건1> 1~4 칼럼 대상 vector 생성(col1, col2, col3, col4)    
   조건2> 1,4 칼럼 대상 합계, 평균, 표준편차 구하기 
   조건3> 1,2 칼럼과 3,4 칼럼을 대상으로 각 df1, df2 데이터프레임 생성
   조건4> df1과 df2 칼럼 단위 결합 iris_df 데이터프레임 생성      
'''

import pandas as pd

iris = pd.read_csv('C:/ITWILL/4_Python_ML/data/iris.csv')
print(iris.info()) # (150, 5)

cols = list(iris.columns)
print(cols)

# 조건1
col1 = iris[cols[0]]
col2 = iris[cols[1]]
col3 = iris[cols[2]]
col4 = iris[cols[3]]

# 조건2
col5 = pd.concat([col1, col4], axis=1)
print(col5)
print(col5.sum())
print(col5.mean())
print(col5.std())

# 조건3
df1 = pd.concat([col1,col2], axis=1)
df2 = pd.concat([col3,col4], axis=1)

# 조건
col6 = pd.concat([df1,df2], axis=1)
print(col6) # (150, 4)

