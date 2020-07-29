'''
문02) winequality-both.csv 데이터셋을 이용하여 다음과 같이 처리하시오.
   <조건1> quality, type 칼럼으로 교차분할표 작성 
   <조건2> 교차분할표를 대상으로 white 와인 내림차순 정렬       
   <조건3> red 와인과 white 와인의 quality에 대한 두 집단 평균 검정
           -> 각 집단 평균 통계량 출력
   <조건4> alcohol 칼럼과 다른 칼럼 간의 상관계수 출력  
'''

import pandas as pd
from scipy import stats

winequality = pd.read_csv("C:/ITWILL/4_Python-II/data/winequality-both.csv")
winequality.info()

# 조건1
sub1 = winequality[['quality', 'type']]
qua = sub1.quality
typ = sub1.type
table = pd.crosstab(qua, typ)
table
# 조건2
table.sort_values('white', ascending=False)
# 조건3
red = sub1[sub1['type']=='red']
white = sub1[sub1['type']=='white']
red_qu = red.quality
white_qu = white.quality
two_sample = stats.ttest_ind(red_qu, white_qu)
two_sample # statistic=-9.685649554187696, pvalue=4.888069044201508e-22
# 귀무가설 기각
# 대립가설 채택 -> 단측 검정
red_qu.mean() # 5.6360225140712945
white_qu.mean() # 5.87790935075541
# white 와인 품질이 더 좋다.


# 조건4
winequality.alcohol.corr(winequality.quality) # 0.4443185200075178
winequality.alcohol.corr(winequality.density) # -0.6867454216813402
cor = winequality.corr()
cor['alcohol']



