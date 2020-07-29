# -*- coding: utf-8 -*-
"""
step03_reshape

1. image shape : 3차원(세로, 가로, 컬러) <컬러가 3이면 RGB>
2. reshape : size 변경 안됨
    ex) [2, 5] -> [5, 2] (o)
        [3, 4] -> [4, 2] (x)
"""
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits # 데이터셋 제공



# 1. image shape


# 1-1) image 가져오기
file_path = "C:/ITWILL/4_Python-II/workspace/chap04_Numpy/images/test1.jpg"
image = imread(file_path) # 이미지 읽기


# 1-2) image 정보
image # 3차원 array구조
type(image) # numpy.ndarray 
image.shape # (360, 540, 3) -> (세로, 가로, 컬러)


# 1-3) image 출력하기
plt.imshow(image)
# 컬러부분이 열로 3이면 red면,green면,blue면 이 겹겹이 겹쳐서 색을만듬


# 1-4) RGB색상 분류
image[0,0,:] # [93, 93, 85] 93=R, 93=G, 85=B
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]
r.shape # (360, 540)
g.shape # (360, 540)
b.shape # (360, 540)



# 2. image data reshape


# 2-1) digit

# 2-1-1) digits dataset 가져오기
digit = load_digits() # dataset loading
digit.DESCR # 설명보기

# 2-1-2) x,y변수 생성
x = digit.data # x변수(입력변수) : image
y = digit.target # y변수(정답=정수)
x.shape # (1797, 64) #(행,열) < 64 -> 8x8 >
y.shape # (1797,)

# 2-1-3) image 생성
x[0] # 1행 index 64개
img_0 = x[0].reshape(8,8) # reshape 을 해주어야 함<3차원>
img_0.shape # (8, 8)
img_0
plt.imshow(img_0) # 0처럼보임
plt.imshow(x[1].reshape(8,8)) # 1로 보임 

# 2-1-4) 정답확인
y # <1차원> array([0, 1, 2, ..., 8, 9, 8])
y[0] # 0
y[1] # 1


# 2-2) reshape
# (전체이미지, 세로, 가로, 컬러(생략시흑백))

# 2-2-1) 3차원(흑백)
x_3d = x.reshape(-1, 8, 8)
x_3d.shape # (1797, 8, 8) -> 흑백이미지 

# 2-2-2) 3차원(흑백) -> 4차원(흑백)
# np.newaxis : 새로운 축 추가
x_4d = x_3d[:, :, :, np.newaxis] # 4번축 추가
x_4d.shape # (1798, 8, 8, 1) -> 흑백이미지



# 3. reshape
''' 전치행렬 : T
    swapaxis = 전치행렬
    transpose() : 3차원 이상 모양 변경  '''


# 3-1) 전치행렬
data = np.arange(10).reshape(2, 5)
data.shape # (2, 5)
data
''' array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])  '''
print(data.T) # 전치행렬
''' [[0 5]
     [1 6]
     [2 7]
     [3 8]
     [4 9]] '''
    

# 3-2) transpose()
''' 1차원 : 효과없음
    2차원 : 전치행렬과 동일
    3차원 : (0, 1, 2) -> 변경하고자하는모양(a, b, c) '''
    
# 3-2-1) dataset 생성
arr3d = np.arange(1, 25).reshape(4, 2, 3)
arr3d.shape # (4, 2, 3)
arr3d
''' array([[[ 1,  2,  3],
            [ 4,  5,  6]],

           [[ 7,  8,  9],
            [10, 11, 12]],

           [[13, 14, 15],
            [16, 17, 18]],
           
           [[19, 20, 21],
            [22, 23, 24]]])   '''

# 3-2-2) reshape (0,1,2) -> (2,1,0)  <0과2 위치바꿈>
arr3d_tran = arr3d.transpose(2,1,0)
arr3d_tran.shape # (3, 2, 4)
arr3d_tran

# 3-2-3) reshape (0,1,2) -> (1,2,0)  <0,1,2 위치다바꿈>
arr3d_tran2 = arr3d.transpose(1,2,0)
arr3d_tran2.shape # (2, 3, 4)
arr3d_tran2















