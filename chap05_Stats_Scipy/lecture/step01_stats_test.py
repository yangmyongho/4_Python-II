# -*- coding: utf-8 -*-
"""
step01_stats_test

scipy 패키지의 확률분포 검정
1. 정규분포 검정 : 연속변수의 확률분포
    - 연속확률분포 : 정규분포, 균등분포, 카이제곱, T/Z/F분포
2. 이항분포 검정 : 2가지(성공/실패) 분포의 확률분포
    - 이산확률분포 : 이항분포, 베르누이분포, 포아송분포
"""
from scipy import stats # 확률분포 검정
import numpy as np
import matplotlib.pyplot as plt # 히스토그램



# 1. 정규분포 검정 : 평균 중심 좌우 대칭성 검정

# 1-1) 정규분포 객체 생성
mu = 0 ; std = 1 # 표준 정규분포
std_norm = stats.norm(mu, std) # 정규분포객체 생성
std_norm # object info


# 1-2) 정규분포확률변수
N = 1000
norm_data = std_norm.rvs(N) # 시뮬레이션 : 1000개 난수 생성 <rvs(n): n번 시행>
len(norm_data) # 1000


# 1-3) 히스토그램 : 좌우대칭성 확인 
plt.hist(norm_data)
plt.show()


# 1-4) 정규성 검정
# 귀무가설(H0) : 정규분포와 차이가 없다. (채택)
# 대립가설() : 정규분포와 차이가 있다.
stats.shapiro(norm_data) 
''' (검정통계량 : 0.99836266040802, pvalue : 0.4682895243167877) '''
test_stats, pvalue = stats.shapiro(norm_data) 
print('검정통계량 : %.5f'%test_stats) # 검정통계량 : 0.99836 -> -1.96 ~ +1.96 : 채택역
print('pvalue : %.5f'%pvalue) # pvalue : 0.46829 >= 알파(0.05) : 채택역



# 2. 이항분포 검정 : 2가지 (성공/실패) 범주의 확률분포 + 가설검정
''' 
 - 베르누이 분포 : 이항변수(성공or실패)에서 성공(1)이 나올 확률분포(모수:성공확률)
 - 이항분포 : 베르누이 분포 에 시행횟수(N)을 적용한 확률분포(모수:P, N) 
ex) P = 게임에 이길확률(40%), N = 시행횟수(100) -> 성공횟수(?) '''
N = 100 # (시행횟수)
P = 0.4 # (성공확률)


# 2-1) 베르누이 분포 -> 이항분포 확률변수
X = stats.bernoulli(P).rvs(N) # 성공확률40% -> 100번 시뮬레이션
X # 0(실패) or 1(성공)


# 2-2) 성공횟수
X_succ = np.count_nonzero(X)
X_succ # 43
print('성공횟수 :', X_succ) # 성공횟수 : 43


# 2-3) 이항분포 검정 : 이항분포에 대한 가설검정
''' 귀무가설 : 게임에 이길 확률은 40%와 다르지 않다.
    대립가설 : 게임에 이길 확률은 40%와 다르다. 
    
stats.binom_test(x, n, p=0.5, alternative='two-sided')
    x : 성공횟수
    n : 시행횟수
    p : 성공확률
    alternative='two-sided' : 양측검정 '''
pvalue = stats.binom_test(x=X_succ, n=N, p=P, alternative='two-sided') 
pvalue # 0.5418708442462032
if pvalue >= 0.05 : # 채택역
    print('게임에 이길 확률은 40%와 다르지 않다.')
else:
    print("게임에 이길 확률은 40%와 다르다.")
# 게임에 이길 확률은 40%와 다르지 않다.


# 2-4) 만약 성공횟수가 25인경우 
pvalue2 = stats.binom_test(x=25, n=N, p=P, alternative='two-sided')
pvalue2 # 0.002070808679256288
if pvalue2 >= 0.05 :
    print('게임에 이길 확률은 40%와 다르지 않다.')
else:
    print("게임에 이길 확률은 40%와 다르다.")
# 게임에 이길 확률은 40%와 다르다.



# 3. 문제1) 
''' 100명의 합격자 중에서 남자 합격자는 45명일때 
    남여 합격률에 차이가 있다고 할수있는가? '''
# 귀무가설 : 남여 합격률에 차이가 없다. (p=0.5)
x = 45
n = 100
p = 0.5
pvalue3 = stats.binom_test(x=45, n=100, p=0.5, alternative='two-sided')
pvalue3 # 0.36820161732669654
# 귀무가설 채택 : 남여 합격률에 차이가 없다.























