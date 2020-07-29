# -*- coding: utf-8 -*-
"""
Zeros Matrix -> Sparse Matrix
문) 영행렬(zero matrix) 이용하여 희소행렬(sparse matrix) 만들기
    <조건1> 단계1과 단계2는 문장과 단어를 만드는 단계(작업이 왼성되었음)
    <조건2> 단계3 부터 문제를 해결하시오.
"""
import numpy as np
import pandas as pd


## 단계1 : 다음 texts를 대상으로 줄 단위로 5개 문장(stances) 만들기 
texts = """우리나라 대한민국 우리나라 대한민국 만세
비아그라 정력 최고
나는 대한민국 사람
보험료 평생 보장 마감 임박
나는 홍길동"""

tokens = texts.split('\n') # 줄 단위로 토큰 생성 
print(tokens)
'''
['우리나라 대한민국 우리나라 만세', '비아그라 정력 최고', '나는 대한민국 사람', '보험료 평생 보장 마감 임박', '나는 홍길동']
'''
# [해설] texts를 대상으로 줄 단위로 문자열을 생성하기 위해서 줄바꿈 기호('\n')를 이용한다.  


## 단계2 : 문장과 단어 만들기 
sentens = tokens
words = []
for st in sentens : # 문장 만들기 
    for word in st.split() : # 단어 만들기  
        words.append(word)
        
    
print('문장 개수 =', len(sentens)) # 문장 개수 = 5
print('단어 개수 =', len(words)) # 단어 개수 = 18
# [해설] 단계1에서 생성한 tokens를 대상으로 문장(stantens)과 문장을 구성하는 단어(words) 생성


## 단계3 : 영행렬(zeros matrix) 만들기
# [설명] 단계2에서 생성한 문장 개수 만큼 행 크기, '중복되지 않은' 단어 개수 만큼 열 크기로 영행렬 생성
words2 = pd.unique(words)
zeros_matrix = np.zeros((len(sentens), len(words2)))
zeros_matrix

# [영행렬(5x14) 완성 결과화면] 
'''
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
'''


## 단계4 : 데이터프레임 만들기 
# [설명] 단계3에서 만든 영행렬을 대상으로 각 열에 중복되지 않은 단어를 열 이름으로 지정
zeros_matrix = pd.DataFrame(zeros_matrix, columns=words2)
zeros_matrix
# [데이터프레임(5x14) 결과화면] 
'''
   대한민국   만세   보장   최고  보험료   임박   나는   정력   마감  홍길동  비아그라   사람   평생  우리나라
0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0   0.0
1   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0   0.0
2   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0   0.0
3   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0   0.0
4   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0   0.0
'''

# 단계5 : 희소행렬 만들기 
# [설명] 단계4에서 만든 DataFrame을 대상으로 각 문서에서 출현한 단어의 수 만큼 셀에 채우기
for j in range(14) :
    x = words2[j]==zeros_matrix.columns[j]
    print(x)
words2[0] # 우리나라
zeros_matrix.columns[0] # 우리나라
words2[0]==zeros_matrix.columns[0]

cnt = 1
nnt = 0
zero = zeros_matrix.copy()
'''
for i in range(5):
    for j in range(14):
        z = words2[j]==zeros_matrix.columns[j]
        if z:
            zero[]
        else:
            zero[i, j] = nnt
zero[0, :]     
'''


# [희소행렬(5x14) 결과화면] 
'''
   대한민국   만세   보장   최고  보험료   임박   나는   정력   마감  홍길동  비아그라   사람   평생  우리나라
0   2.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0   2.0
1   0.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0   1.0  0.0  0.0   0.0
2   1.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0   0.0  1.0  0.0   0.0
3   0.0  0.0  1.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0   0.0  0.0  1.0   0.0
4   0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0   0.0  0.0  0.0   0.0
'''
#[해설] 첫번째 문장에서 '대한민국'과 '우리나라' 단어는 2회씩 출현 


