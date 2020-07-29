'''
문1) iris.csv 파일을 이용하여 다음과 같이 차트를 그리시오.
    <조건1> iris.csv 파일을 iris 변수명으로 가져온 후 파일 정보 보기
    <조건2> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그리기
    <조건3> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그린 후  5번 칼럼으로 색상 적용 
'''

import pandas as pd
import matplotlib.pyplot as plt

# 조건1
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")
iris1 = iris.iloc[:,0]
iris2 = iris.iloc[:,2]
iris3 = iris.iloc[:,4]
iris3.unique() # array(['setosa', 'versicolor', 'virginica'], dtype=object)

# 조건2
plt.plot(iris1, 'ro')
plt.plot(iris2, 'b^')
plt.show()
plt.scatter(iris1, iris2)
plt.show()

# 조건3
sp = []
for s in iris3:
    if s == 'setosa':
        sp.append(1)
    elif s == 'versicolor':
        sp.append(2)
    else:
        sp.append(3)

plt.scatter(iris1, iris2, c=sp, marker='o')
plt.show()








