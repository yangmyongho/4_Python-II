# -*- coding: utf-8 -*-
"""
step05_dot_example

신경망에서 행렬곱 적용 예) 
    - 은닉층(h) = [입력(X) * 가중치(w)] + 편향(b)
"""
import numpy as np



# 1. ANN model
''' image(28x28) <컬러생략시 흑백>
    hidden node : 32개
    weight = [? , ?]   ->   [28x32]  '''
    
# 2. input data : image data
28 * 28 # 784
x_img = np.random.randint(0, 256, size=784) # 0 ~ 255 
x_img.shape # (784,)
x_img.max() # 255

# 이미지 정규화 : 0 ~ 1 사이
x_img = x_img / 255
x_img.shape # (784,)
x_img
x_img = x_img.reshape(28, 28)
x_img.shape # (28, 28)
x_img

# 3. weight data : 
weight = np.random.randn(28,32)
weight.shape # (28, 32)
weight
    
# 4. hidden layer
hidden = np.dot(x_img, weight)
hidden.shape # (28, 32) = x(28, 28) * w(28, 32)
hidden
























































