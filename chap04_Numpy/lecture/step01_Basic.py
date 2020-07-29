# -*- coding: utf-8 -*-
"""
step01_Basic

Numpy 패키지 특징
 - 선형대수(벡터, 행렬) 연산 관련 함수 제공
 - list 보다 이점 : N차원 배열 생성, 선형대수 연산, 고속 처리(for문 사용x)
 - Series 공통점
     -> 수학/통계 함수 지원
        ex) obj.수학/통계()
     -> 범위수정, 블럭연산가능
     -> indexing/slicing 가능
 - 주요 모듈 함수
     1. random : 난수 생성 함수
     2. array 함수 : N차원 배열 생성(array([[list]])) <중첩갯수만큼 중첩list생성>
     3. sampling 함수 
     4. arrange 함수 : : range() 유사함

참고 사이트 : https://www.scipy.org
"""
import numpy as np
import pandas as pd



# 1. list 와 numpy


# 1-1) list 자료 구조
lst = [1, 2, 3]
#lst**2 # 오류 : list에서는 연산 할수없음 
for i in lst:
    print(i**2)
lst*2 # [1, 2, 3, 1, 2, 3] 숫자연산이 아닌 반복


# 1-2) numpy 자료 구조
arr = np.array(lst) # list -> numpy
arr # array([1, 2, 3])
arr**2 # array([1, 4, 9], dtype=int32)


# 1-3) numpy 특성 : 동일 type
arr2 = np.array([1,2,3]) # 숫자형(연산가능)
arr2 # array([1, 2, 3])
arr3 = np.array([1,'two',3]) # 문자형(연산불가능)
arr3 # array(['1', 'two', '3'], dtype='<U11')
arr3.shape # (3,)
arr4 = np.array([[1,'two',3]]) # 중첩list 사용 -> 2차원
arr4 # array([['1', 'two', '3']], dtype='<U11')
arr4.shape # (1, 3)



# 2. random : 난수 생성 함수
data = np.random.randn(3,4) # 형식) 모듈.모듈.함수(행,열) : 2차원
data.shape # (3, 4)
data
''' array([[-2.22156104,  0.16202913, -2.16037839,  0.73182739],
           [-1.37966417, -0.38193414, -0.8285389 , -0.14225854],
           [ 0.67105854,  1.21336954, -1.57172509, -0.76287643]])  '''

    
# 2-1) for 문 사용시 행단위로 사용함
for row in data:
    print('행 단위 합계 :', row.sum())
    print('행 단위 평균 :', row.mean())
''' 행 단위 합계 : -3.4880829140538023
    행 단위 평균 : -0.8720207285134506
    행 단위 합계 : -2.732395762155588
    행 단위 평균 : -0.683098940538897
    행 단위 합계 : -0.4501734345108971
    행 단위 평균 : -0.11254335862772427  '''


# 2-2) 수학/통계 함수 지원 
type(data) # numpy.ndarray
# 형식) obj.수학/통계()
# 2-2-1) 기본 함수
print('전체 합계 :', data.sum()) # 전체 합계 : -6.670652110720288
print('전체 평균 :', data.mean()) # 전체 평균 : -0.5558876758933573
print('전체 분산 :', data.var()) # 전체 분산 : 1.1822671876532747
print('전체 표준편차 :', data.std()) # 전체 표준편차 : 1.087321106046082

# 2-2-2) 기타 사용 가능 함수
dir(data) # 사용가능한 함수 검색
data.shape # (3, 4)
data.size # 12


# 2-3) 범위수정, 블럭연산
# 2-3-1) 범위수정
data*2 # 2배

# 2-3-2) 블럭연산
data + data # 2배
data - data # 0 <3행4열구조는 유지>


# 2-4) indexing : R 과 유사함
data[0,0] # -2.221561042642396 <1행1열>
data[0,:] # 1행 전체
data[:,1] # 2열 전체



# 3. array 함수 : N차원 배열 생성


# 3-1) 단일 list
lst1 = [3, 5.6, 4, 7, 8]
lst1 # [3, 5.6, 4, 7, 8]
arr1 = np.array(lst1) # list -> numpy
arr1 # array([3. , 5.6, 4. , 7. , 8. ]) <동일 type으로 맞춰진다.>
arr1.var() # 3.4016000000000006 분산
arr1.std() # 1.8443427013437608 표준편차


# 3-2) 중첩 list
lst2 = [[1,2,3,4,5], [2,3,4,5,6]]
lst2 # [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
lst2[0] # [1, 2, 3, 4, 5] 
lst2[0][2] # 3
arr2 = np.array(lst2)
arr2
''' array([[1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6]])  '''
arr2.shape # (2, 5)  2차원
arr2[0,:] # array([1, 2, 3, 4, 5]) <1행전체>
arr2[0,2] # 3


# 3-3) indexing : obj[행index, 열index]
arr2[1,:] # array([2, 3, 4, 5, 6]) <2행전체>
arr2[:,1] # array([2, 3]) <2열전체>
arr2[:, 2:4]
''' array([[3, 4],
           [4, 5]])  <1,2행 3,4열> 박스형구조  '''

    
# 3-4) brodcast 연산 : 작은 차원이 큰 차원으로 늘어난 후 연산 
# 3-4-1) scala(0)  vs  vector(1)
arr1 # array([3. , 5.6, 4. , 7. , 8. ])
0.5*arr1 # array([1.5, 2.8, 2. , 3.5, 4. ])

# 3-4-2) scala(0)  vs  matrix(2)
arr2
''' array([[1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6]]) '''
0.5 * arr2
''' array([[0.5, 1. , 1.5, 2. , 2.5],
           [1. , 1.5, 2. , 2.5, 3. ]])  '''
    
# 3-4-3) vaetor(1)  vs  matrix(2)
print(arr1.shape, arr2.shape) # (5,) (2, 5)
arr3 = arr1 + arr2 
arr3
''' array([[ 4. ,  7.6,  7. , 11. , 13. ],
           [ 5. ,  8.6,  8. , 12. , 14. ]]) '''
arr1 * arr2
''' array([[ 3. , 11.2, 12. , 28. , 40. ],
           [ 6. , 16.8, 16. , 35. , 48. ]]) '''


    
# 4. sampling 함수
num = list(range(1,11))
num  #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# 4-1) sampling 함수 개념
help(np.random.choice)
''' a, size=None, replace=True, p=None
        a : 관측치 길이
        size : 임의 추출 크기
        replace : 복원(True) or 비복원(False)
        p : 확률                             '''
idx = np.random.choice(a=len(num), size=5, replace=False) # 비복원 추출
idx # array([4, 9, 1, 7, 6])


# 4-2) score_iq.csv 적용
# 4-2-1) score dataset 생성
score = pd.read_csv("C:/ITWILL/4_Python-II/data/score_iq.csv")
score
score.shape # (150, 6)
len(score) # 150

# 4-2-2) random.choie 
idx2 = np.random.choice(a=len(score), size=int(len(score)*0.3), replace=False)
idx2
len(idx2) # 45

# 4-2-3) train셋 생성(DataFrame)
score_train = score.iloc[idx2, :] # DataFrame index <iloc>
score_train.shape # (45, 6)

# 4-2-4) pandas(DF) -> numpy(array)
score_arr = np.array(score)
score_arr.shape # (150, 6)

# 4-2-5) train셋 생성(array)
score_train2 = score_arr[idx2, :]
score_train2.shape # (45, 6)

# 4-2-6) test셋 생성
test_id = [i for i in range(len(score)) if i not in idx2]
# 마저 하




# 5. arrange 함수 : : range() 유사함


# 5-1) zero 행렬 생성
zero_arr = np.zeros((3, 5))
zero_arr # 모든값이 0인 3행5열


# 5-2) zero 행렬 값 채우기(range)
cnt = 1
for i in range(3): # 행 index
    for j in range(5): # 열 index
        zero_arr[i,j] = cnt
        cnt += 1
zero_arr
''' array([[ 1.,  2.,  3.,  4.,  5.],
           [ 6.,  7.,  8.,  9., 10.],
           [11., 12., 13., 14., 15.]])  '''
#range(-1.0, 2, 0.1) # (start, stop, step)
# range는 음수,소수 사용불가능
    
    
# 5-3) zero 행렬 값 채우기(arange)
cnt = 1
for i in np.arange(3): # 행 index
    for j in np.arange(5): # 열 index
        zero_arr[i,j] = cnt
        cnt += 1
zero_arr
''' array([[ 1.,  2.,  3.,  4.,  5.],
           [ 6.,  7.,  8.,  9., 10.],
           [11., 12., 13., 14., 15.]])  ''' # 결과 range와 같음 
np.arange(-1.0, 2, 0.1) # (start, stop, step) 
np.arange(-1.0, 2, 0.1)
# arange는 음수,소수도 사용가능


















