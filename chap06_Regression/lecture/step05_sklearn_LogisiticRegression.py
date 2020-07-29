# -*- coding: utf-8 -*-
"""
step05_sklearn_LogisiticRegression

sklearn 로지스틱 회귀모델
 - y변수가 범주형인 경우 사용가능

"""
from sklearn.datasets import load_breast_cancer, load_iris, load_digits # dataset load
from sklearn.linear_model import LogisticRegression # model 생성
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn # heatmap - Accuracy Score
from sklearn.model_selection import train_test_split



# 1. 이항분류모델 (breast_cancer)


# 1-1) dataset load & 변수선택
breast = load_breast_cancer()
X = breast.data
y = breast.target
X.shape # (569, 30)  2차원
y.shape # (569,)     1차원


# 1-2) model 생성
# 1-2-1) LogisticRegression 에 대해서
help(LogisticRegression)
''' random_state=None, solver='lbfgs', max_iter=100, multi_class='auto'

random_state=None : 난수 seed값 지정
solver='lbfgs' : 알고리즘
    ㄴ> {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
max_iter=100 : 반복학습 횟수
multi_class='auto' : 다항분류 
    ㄴ> {'auto', 'ovr', 'multinomial'}, default='auto'  
    
적용 예)
일반 데이터, 이항분류 : defalt
일반 데이터, 다항분류 : multi_class=multinomial
빅 데이터, 이항분류 : solver='sag' or 'saga'
빅 데이터, 다항분류 : solver='sag' or 'saga' , multi_class=multinomial

활성함수
sigmoid function : 0 ~ 1 확률값 -> cutoff=0.5 -> 이항분류
softmax function : 0 ~ 1 확률값 -> 전체합=1(c0:0.1, c1:0.2, c2:0.7) -> 다항분류
'''

# 1-2-2) model 생성
''' 일반데이터, 이항분류 : default '''
obj = LogisticRegression(random_state=123)
model = obj.fit(X=X, y=y) 
# multi_class='auto' -> sigmoid 활용함수 이용 -> 이항분류
y_pred = model.predict(X)


# 1-3) model 평가
# 1-3-1) .score 사용
acc = model.score(X, y) # 예측치를 구할 필요가 없음 
print('accuracy =', acc) # accuracy = 0.9472759226713533 <분류정확도>
# 1-3-2) accuracy_score 사용
acc2 = accuracy_score(y, y_pred)
print('accuracy =', acc2) # accuracy = 0.9472759226713533 <동일한 결과>
# 1-3-3) confusion_matrix + 분류정확도 구하는식 1 사용
con_mat = confusion_matrix(y, y_pred) # 교차분할표형식
print(con_mat)
'''       0   1
   0  [[193  19]
   1   [ 11 346]]   '''
type(con_mat) # numpy.ndarray
acc3 = (con_mat[0,0]+con_mat[1,1])  /  con_mat.sum()
print('accuracy =', acc3) # accuracy = 0.9472759226713533  <동일한 결과>
# 1-3-5) 분류정확도 구하는식2 사용
tab = pd.crosstab(y, y_pred, rownames=['관측치'], colnames=['예측치'])
tab
''' 예측치    0    1
    관측치          
      0    193   19
      1     11  346         '''
acc4 = (tab.loc[0,0] + tab.loc[1,1]) / len(y)
print('accuracy =', acc4) # accuracy = 0.9472759226713533  <동일한 결과>



# 2. 다항분류모델 (iris)


# 2-1) dataset load & X,y변수선택
iris = load_iris()
iris.target_names # ['setosa', 'versicolor', 'virginica']
X, y = load_iris(return_X_y=True)
X.shape # (150, 4)
y.shape # (150,)
y # 0 ~ 2


# 2-2) model 생성
''' 일반 데이터, 다항분류 : multi_class=multinomial '''
obj = LogisticRegression(random_state=123, multi_class='multinomial')
model = obj.fit(X=X, y=y) 
# multi_class='multinomial' -> softmax 활용함수 이용 -> 다항분류

# 2-2-1) 예측값 생성
y_pred = model.predict(X) # class
y_pred2 = model.predict_proba(X) # 확률값
y_pred # 0 ~ 2 예측값
y_pred2 # 예측한 확률값
y_pred.shape # (150,)
y_pred2.shape # (150, 3)

# 2-2-2) y_pred  vs  y_pred2 
y_pred[0] # 0     < 0이 나올거라고 예측 >
y_pred2[0] # [9.81797141e-01, 1.82028445e-02, 1.44269293e-08] < 0 , 1, 2 값이 나올 확률 >
arr = np.array([9.81797141e-01, 1.82028445e-02, 1.44269293e-08])
arr.max() # 0.981797141
arr.min() # 1.44269293e-08
arr.sum() # 0.9999999999269293  <1에 수렴한다.>


# 2-3) model 평가   <이때 y_pred(예측값)을 넣어야 한다.>
# 2-3-1) accuracy_score 사용
acc = accuracy_score(y, y_pred)
print('accuracy =', acc) # accuracy = 0.9733333333333334 <분류정확도>
# 2-3-2) confusion_matrix + 분류정확도 구하는식 1 사용
con_mat = confusion_matrix(y, y_pred)
con_mat
'''   [[50,  0,  0],
       [ 0, 47,  3],
       [ 0,  1, 49]]     '''
acc2 = (con_mat[0,0] + con_mat[1,1] + con_mat[2,2]) / con_mat.sum()
print('accuracy =', acc2) # accuracy = 0.9733333333333334


# 2-4) 히트맵 시각화 (confusion matrix heatmap )
plt.figure(figsize=(6,6)) # chart size
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square = True);# , cmap = 'Blues_r' : map »ö»ó 
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 18)
plt.show()



# 3. 다항분류모델 (digits)


# 3-1) dataset load
digits = load_digits()
digits.target_names # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
X = digits.data
y = digits.target
X.shape # (1797, 64)  -> 1797장 images
y.shape # (1797,)  -> 1797장 images 10진수 정답값


# 3-2) data split
img_train, img_test, label_train, label_test = train_test_split(X, y, 
                                                                random_state=123)
img_train.shape # (1347, 64)
img_test.shape # (450, 64)

# 3-2-1) 훈련셋(train) image  ->  reshape(8, 8)
img2d = img_train.reshape(-1,8,8) # (전체image, 세로, 가로)
img2d.shape # (1347, 8, 8)
plt.imshow(img2d[0]) # 3처럼보이지만...7이다.
label_train[0] # 7


# 3-3) model 생성
obj = LogisticRegression(multi_class='multinomial')
model = obj.fit(img_train, label_train)
y_pred = model.predict(img_test)
y_pred2 = model.predict_proba(img_test)
y_pred
y_pred2


# 3-4) model 평가
# 3-4-1) accuracy_score
acc1 = accuracy_score(label_test, y_pred)
acc1 # 0.9644444444444444

# 3-4-2) confusion_matrix
con_mat = confusion_matrix(label_test, y_pred)
con_mat
'''
array([[51,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0, 42,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, 41,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 39,  0,  0,  0,  0,  0,  1],
       [ 0,  0,  0,  0, 51,  0,  0,  1,  0,  0],
       [ 0,  1,  0,  0,  0, 42,  0,  2,  0,  3],
       [ 0,  1,  0,  0,  0,  1, 46,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 41,  0,  0],
       [ 0,  4,  0,  0,  0,  0,  0,  0, 41,  0],
       [ 0,  1,  0,  0,  0,  0,  0,  0,  1, 40]], dtype=int64)   '''


# 3-5) result 확인
result = label_test  == y_pred
result # 일치 여부
result.mean() # 0.96444444444444
len(result) #450

# 3-5-1) 틀린 image
false_img = img_test[result == False] # 위아래 결과 같음
false_img.shape # (16, 64)
false_img3d = false_img.reshape(-1, 8, 8)
false_img3d.shape # (16, 8, 8)

for idx in range(false_img3d.shape[0]): # 행 단위
    print(idx)
    plt.imshow(false_img3d[idx])
    plt.show()

# 3-6) 히트맵 시각화 (confusion matrix heatmap )
plt.figure(figsize=(6,6)) # chart size
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square = True);# , cmap = 'Blues_r' : map »ö»ó 
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc1)
plt.title(all_sample_title, size = 18)
plt.show()

























