# -*- coding: utf-8 -*-
"""
step02_indexing

indexing/slicing
 - 1차원 indexing : list 와 동일함
 - 2,3차원 indexing
 - boolean indexing
"""
import numpy as np
import pandas as pd
''' 1차원 : obj[index]
    2차원 : obj[행index, 열index]
    3차원 : obj[면index, 행index, 열index] '''
    


# 1. 1차원 indexing
    
    
# 1-1) list 객체
ldata = [0, 1, 2, 3, 4, 5]
ldata # [0, 1, 2, 3, 4, 5]
ldata[:] # [0, 1, 2, 3, 4, 5] <전체원소>
ldata[2:] # [2, 3, 4, 5] <[n:~]>
ldata[:3] # [0, 1, 2] <[~:n]
ldata[-1] # 5
#ldata[:] = 10 <list 는 블럭수정불가능>


# 1-2) numpy 객체
arrld = np.array(ldata)
arrld # array([0, 1, 2, 3, 4, 5])
arrld.shape # (6,)
arrld[2:] # array([2, 3, 4, 5])
arrld[:3] # array([0, 1, 2])
arrld[-1] # 5


# 1-3) slicing
arr = np.array(range(1,11))
arr # <원본> array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
arr_sl = arr[4:8]
arr_sl # <사본> array([5, 6, 7, 8])


# 1-4) 블럭수정
arr_sl[:] = 50
arr_sl # array([50, 50, 50, 50])
arr # <원본수정> array([ 1,  2,  3,  4, 50, 50, 50, 50,  9, 10])



# 2. 2차원 indexing : [행, 열]


# 2-1) numpy객체 생성
arr2d = np.array([[1,2,3], [2,3,4], [3,4,5]])
arr2d.shape # (3, 3)
arr2d
''' array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])  '''


# 2-2) 행index : default
arr2d[1] # = arr2d[1, :]  ->  array([2, 3, 4]) <2행전체>
arr2d[1:] # 2~3행
arr2d[:, 1:] # 2~3열
arr2d[2, 1] # 4  <3행2열>
arr2d[:2, 1:] # 박스선택
''' array([[2, 3],
           [3, 4]])   '''



# 3. 3차원 indexing : [면, 행, 열]


# 3-1) numpy객체 생성
arr3d = np.array([ [[1,2,3,4],[2,3,4,5],[3,4,5,6]], 
                  [[4,5,6,7],[5,6,7,8],[6,7,8,9]] ]) # (2,3,4) <2면 3행 4열> 
arr3d
'''
array([[[1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]],

       [[4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9]]])
'''

arr3d.shape # (2,3,4)


# 3-2) index

# 3-2-1) 면 index : default
arr3d[0] # 면 index = 1면
arr3d[1] # 2면

# 3-2-2) 면 -> 행 index
arr3d[0, 2] # [3, 4, 5, 6]

# 3-2-3) 면 -> 행 -> 열index
arr3d[1,2,3] # 9
arr3d[1, 1: , 1:] # 박스선택
''' array([[6, 7, 8],
           [7, 8, 9]])   '''



# 4. boolean indexing


# 4-1) numpy객체 생성
dataset = np.random.randint(1, 10, size=100) # 1~100
len(dataset) # 100
dataset


# 4-2) 5이상 자료 선택
dataset2 = dataset[dataset >= 5] # 관계식 사용가능
len(dataset2) # 57
dataset2


# 4-3) 5이상 8이하 자료 선택 : 논리연산자 사용
#dataset[dataset >= 5 and dataset <= 8] # <오류> 논리식과 관계식 함께 직접 사용불가
np.logical_and
np.logical_or
np.logical_not
dataset2 = dataset[np.logical_and(dataset >= 5, dataset <= 8)]
len(dataset2) # 39

















