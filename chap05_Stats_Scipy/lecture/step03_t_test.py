# -*- coding: utf-8 -*-
"""
step03_t_test

집단 간 평균차이 검정
 1. 한 집단 평균차이 검정
 2. 두 집단 평균차이 검정
 3. 대응 두 집단 평균차이 검정
"""
from scipy import stats # t 검정
import numpy as np # 숫자 연산
import pandas as pd # file read



# 1. 한 집단 평균차이 검정
''' 대한민국 남자 평균 키(모평균) : 175.5cm
    모집단 -> 표본 추출(30명) '''
sample_data = np.random.uniform(172, 180, size=300)
sample_data
# 기술통계 
sample_data.mean() # 176.16701764452307
# 평균차이 검정
one_group_test = stats.ttest_1samp(sample_data, 175.5)
one_group_test # statistic=5.0762428191651034, pvalue=6.77956677802154e-07
print('statistic = %.5f, pvalue= %.5f'%(one_group_test))
 # statistic = 5.07624, pvalue= 0.00000
# pvalue = 0.0000 < 0.05 이므로 채택역 기각 
# 따라서 평균값과 차이가 있다.



# 2. 두 집단 평균차이 검정

# 2-1) 여자 집단과 남자 집단의 평균 성적 차이 유무 
female_score = np.random.uniform(50, 100, size=30)
male_score = np.random.uniform(45, 95, size=30)
female_score
male_score
# 기술통계
female_score.mean() # 77.17261777801505
male_score.mean() # 70.95675079133318
# 평균차이 검정
two_sample = stats.ttest_ind(female_score, male_score)
two_sample # statistic=1.7284700906190322, pvalue=0.08922464144606969
# pvalue = 0.0892 > 0.05 이므로 귀무가설 채택
# 남성과 여성의 평균 점수차이가 없다.


# 2-2) 교육방법에따른 성적 차이 유무
# csv file load
two_sample = pd.read_csv("C:/ITWILL/4_Python-II/data/two_sample.csv")
two_sample.info()
# 필요한칼럼만 추출 
sample_data = two_sample[['method', 'score']]
sample_data
sample_data['method'].value_counts() # 2  120   ,   1  120
# 교육방법에 따른 subset
method1 = sample_data[sample_data['method']==1]
method2 = sample_data[sample_data['method']==2]
# 점수추출
score1 = method1.score
score2 = method2.score
# 결측치 제거 (Na 제거)
score1 = score1.fillna(score1.mean())
score2 = score2.fillna(score2.mean())
# 기술통계
score1.mean() # 5.496590909090908
score2.mean() # 5.591304347826086
# 집단별 검정
two_sample_test = stats.ttest_ind(score1, score2)
two_sample_test # statistic=-0.9468624993102985, pvalue=0.34466920341921115
# pvalue = 0.344 > 0.05 이므로 귀무가설 채택
# 따라서 교육방법에 따른 성적 차이 없다.



# 3. 대응 두 집단 평균차이 검정 : 복용전(65) -> 복용후(60)
before = np.random.randint(65, size=30) * 0.5
after = np.random.randint(60, size=30) * 0.5
before
after
# 기술통계
before.mean() # 16.95
after.mean() # 14.88333
pired_test = stats.ttest_rel(before, after)
pired_test # statistic=0.8109456442862958, pvalue=0.424002635104881
# pvalue = 0.4240 > 0.05 이므로 귀무가설 채택
# 따라서 복용전과 복용후 평균차이 없다.















































