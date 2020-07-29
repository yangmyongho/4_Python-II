# -*- coding: utf-8 -*-
"""
step06_image_resize

reshape  vs  resize
 - reshape : 모양 변경
 - resize : 크기 변경
  ex) images -> 120x150 규격화  ->  model

image 규격화 : 실습
"""
from glob import glob # file 검색 패턴 사용(문자열 경로, *.jpg) 
# glob (문자열경로, *...) <jpg뿐만아니라 ...자리에 확장자를 바꾸면 모두 검색 가능> 
from PIL import Image # image file read <Python Image Library>
import numpy as np
import matplotlib.pyplot as plt # 이미지 시각화



# 1. 1개 image resize 함수 

# 1-1) image file open
path = "./chap04_Numpy"
file = path + "/images/test1.jpg"
img = Image.open(file) # image file read
type(img) # PIL.JpegImagePlugin.JpegImageFile
#img.shape # 오류 : numpy 형식이 아니라서 사용불가능 np를 붙혀서 사용가능
np.shape(img) # (360, 540, 3)
img
plt.imshow(img)


# 1-2) (360, 540, 3) -> (120, 150, 3) 으로 변경 -> (세, 가, 컬)=(h, w, c)
img_re = img.resize( (150, 120) ) # w, h < 세로 가로 순서 바꾼채로 입력> 
type(img_re) # PIL.Image.Image
np.shape(img_re) # (120, 150, 3)
img_re
plt.imshow(img_re) # 픽셀정보가 줄어서 화질선명도도 줄어들었다


# 1-3) PIL -> numpy
img_arr = np.asarray(img) # array 구조로 변경
type(img_arr) # numpy.ndarray
img_arr.shape # (360, 540, 3)
img_arr
plt.imshow(img_arr) # 원본 그대로



# 2. 여러장의 image resize 함수
def imageResize():
    img_h = 120 # 세로 픽셀
    img_w = 150 # 가로 픽셀
    image_resize = [] # 규격화된 image 저장
    
    # 2-1) glob : file 패턴
    for file in glob(path + "/images/" + "*.jpg"): # .jpg 로 끝나는 파일들
        # test1.jpg, 2)test2.jpg, ...
        img = Image.open(file) # image file read
        print(np.shape(img)) # image shape(수정전)
        
        # 2-2) PIL -> resize 
        img = img.resize( (img_w, img_h) ) # w, h
        
        # 2-3) PIL -> numpy
        img_data = np.asarray(img)
        
        # 2-4) resize image save
        image_resize.append(img_data)
        print(file, ':', img_data.shape) # image shape(수정후)
    
    # 2-5) list -> numpy
    return np.array(image_resize) 

# 2-6) 결과확인
image_resize = imageResize()
'''
(360, 540, 3)
./chap04_Numpy/images\test1.jpg : (120, 150, 3)
(332, 250, 3)
./chap04_Numpy/images\test2.jpg : (120, 150, 3)
'''
image_resize.shape # (2, 120, 150, 3) -> (size, h, w, color)  <4차원>

# 2-7) index 
image_resize[0].shape # <첫번째사진 shape> (120, 150, 3)  
image_resize[1].shape # <두번째사진 shape> (120, 150, 3) 

# 2-8) image 보기
plt.imshow(image_resize[0]) # 첫번째 사진 보기
plt.imshow(image_resize[1]) # 두번째 사진 보기


































