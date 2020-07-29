# -*- coding: utf-8 -*-
"""
step03_csv_FileIO
 - 1. csv file read
 - 2. csv file write
 - 3. random sampling
"""
import pandas as pd


# 1. csv file read

# 1-1) 칼럼명 : 특수문자 or 공백 -> 문자 변경 가능 
#               . 이나 띄어쓰기는 잘못인식할수도있기때문
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")
iris.info
iris.columns = iris.columns.str.replace('.','_') # str.replace('변경전','후')
iris.info
iris.Sepal_Length

# 1-2) 칼럼명이 없는 경우 (st.coiumns = 수정값)
st = pd.read_csv("C:/ITWILL/4_Python-II/data/student.csv", header = None)
st #   0    1    2   3
col_names = ['학번', '이름', '키', '몸무게']
st.columns = col_names
st # 학번    이름    키  몸무게

# 1-3) 행이름이 없는 경우 (st.index = 수정값)
idx_names = [1, 2, 3, 4]
st.index = idx_names
st

# 1-4) 칼럼추가  <BMI>
# BMI = 몸무게 / 키**2 (단, 이때 단위는 몸무게:kg , 키:m)
BMI = [st.loc[i+1,'몸무게'] / (st.loc[i+1,'키']*0.01)**2 for i in range(len(st))]
BMI
type(BMI) # list
st['비만도'] = BMI
st
st['비만도'] = st['비만도'].round(2)
st


# 2. csv file write
st.to_csv('C:/ITWILL/4_Python-II/data/student_df.csv', 
          index=None, encoding='utf-8') # index=None : index 저장하지않는다
# 확인
st_df = pd.read_csv("C:/ITWILL/4_Python-II/data/student_df.csv")
st_df


# 3. random sampling
wdbc = pd.read_csv("C:/ITWILL/4_Python-II/data/wdbc_data.csv")
wdbc.info
wdbc_train = wdbc.sample(400)
wdbc_train.shape # (400, 32)
wdbc_train.head() # 순서 랜덤







































