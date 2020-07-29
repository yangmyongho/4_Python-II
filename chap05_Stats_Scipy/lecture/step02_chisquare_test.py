# -*- coding: utf-8 -*-
"""
step02_chisquare_test

카이제곱검정(chisquare test)
 - 일원 카이제곱, 이원 카이제곱
"""
from scipy import stats
import numpy as np
import pandas as pd



# 1. 일원 카이제곱 검정
''' 귀무가설 : 관측치와 기대치는 차이가 없다.(게임에 적합하다.) 
    대립가설 : 관측치와 기대치는 차이가 있다.(게임에 적합하지 않다.)  (채택)  '''

# 1-1) 카이제곱검정
real_data = [4, 6, 17, 16, 8, 9] # 관측치 
exp_data = [10,10,10,10,10,10] # 기대치 
chis = stats.chisquare(real_data, exp_data) 
print('statistic = %.3f, pvalue= %.3f'%(chis)) 
# statistic = 14.200, pvalue = 0.014 <= 0.05 이므로 귀무가설 기각


# 1-2) list -> numpy 공식 확인
''' statistic = 14.200 = k^2 = 기대비율
     χ2 = Σ (관측값 - 기댓값)2 / 기댓값 '''
real_arr = np.array(real_data)
exp_arr = np.array(exp_data)
chis2 = sum((real_arr - exp_arr)**2 / exp_arr)
chis2 # 14.200000000000001



# 2. 이원 카이제곱 검정


# 2-1) 교육수준  vs  흡연유무  독립성 검정
# 귀무가설 : 교육수준과 흡연유무 간의 관련성이 없다. (채택)

# 2-1-1) datset 생성
smoke = pd.read_csv("C:/ITWILL/4_Python-II/data/smoke.csv")
smoke.info()

# 2-1-2) DF -> vector
education = smoke.education
smoking = smoke.smoking

# 2-1-3) cross table
table = pd.crosstab(education, smoking)
table
''' smoking     1   2   3
    education            
    1          51  92  68
    2          22  21   9
    3          43  28  21   
    귀무가설을 기각할만큼 많은 차이는 보이지 않는다.   '''

# 2-1-4) 카이제곱 검정
chis = stats.chisquare(education, smoking)
chis # statistic=347.66666666666663, pvalue=0.5848667941187113
# pvalue = 0.5848667941187113 >= 0.05 이므로 귀무가설 채택


# 2-2) 성별  vs  흡연유무 독립성 검정
# 귀무가설 : 성별과 흡연유무 간의 관련성이 없다. (채택)

# 2-2-1) dataset 생성
tips = pd.read_csv("C:/ITWILL/4_Python-II/data/tips.csv")
tips.info()

# 2-2-2) DF -> vector
gender = tips.sex
smoker = tips.smoker

# 2-2-3) cross table
table2 = pd.crosstab(gender, smoker)
table2
''' smoker  No  Yes
    sex            
    Female  13   15   '''

# 2-2-4) 카이제곱 검정 (실패)
#chis2 = stats.chisquare(gender, smoker) # str(문자타입) -> 오류

# 2-2-5) dummy 생성 : 문자 -> 숫자
# 0 or 1 (x) -> 1(Male) or 2(Female) (o)
gender_dummy = [1 if g == 'Male' else 2 for g in gender]
gender_dummy
# 0 or 1 (x) -> 1(No) or 2(Yes) (o)
smoker_dummy = [1 if s == 'No' else 2 for s in smoker]
smoker_dummy

# 2-2-6) 카이제곱 검정 (성공)
chis2 = stats.chisquare(gender_dummy, smoker_dummy)
chis2 # statistic=84.0, pvalue=1.0
# pvalue=1.0 >= 0.05 이므로 귀무가설 채택









