# -*- coding: utf-8 -*-
"""
step01_groupby
   
DataFrame 객체 대상 그룹화
 - 형식) DF.groupby('집단변수').수학/통계함수()    

"""
import pandas as pd


# 1. DataFrame 생성 및 수정
tips = pd.read_csv("C:/ITWILL/4_Python-II/data/tips.csv")
tips.info()
tips.head()

# 1-1) 새로운 파생변수 : 팁 비율(사칙연산)
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()

# 1-2) 변수 복제 : gender = sex
tips['gender'] = tips['sex']

# 1-3) 변수 제거
del tips['sex']
tips.info()


# 2. 집단변수 1개(gender) -> 전체 칼럼 그룹화
gender_grp = tips.groupby('gender')
gender_grp # object info
# 객체 -> 호출 가능한 멤버 확인
dir(gender_grp) 

# 2-1) 그룹객체.함수() : 숫자변수만 대상
gender_grp.size() # 그룹별 빈도수 Female : 87 , Male : 157
gender_grp.sum() # 그룹별 합계
'''         total_bill     tip  size    tip_pct
    gender                                     
    Female     1570.95  246.51   214  14.484694
    Male       3256.82  485.07   413  24.751136 '''
gender_grp.mean() # 그룹별 평균
'''         total_bill       tip      size   tip_pct
    gender                                          
    Female   18.056897  2.833448  2.459770  0.166491
    Male     20.744076  3.089618  2.630573  0.157651 '''
gender_grp.describe() # 그룹별 요약통계량
'''
       total_bill                       ...   tip_pct                    
            count       mean       std  ...       50%       75%       max
gender                                  ...                              
Female       87.0  18.056897  8.009209  ...  0.155581  0.194266  0.416667
Male        157.0  20.744076  9.246469  ...  0.153492  0.186240  0.710345  '''
gender_grp.boxplot() # 그룹별 요약통계량 그래프화


# 3. 집단변수 1개 -> 특정 칼럼 그룹화
smoker_grp = tips['tip'].groupby(tips['smoker'])
smoker_grp # object info

# 3-1) 그룹객체.함수() : 숫자변수만 대상
smoker_grp.size() # 그룹별 빈도수  No 151 , Yes 93
smoker_grp.mean() # 그룹별 특정변수(tip) 평균  No 2.991854 , Yes 3.008710


# 4. 집단변수 2개 -> 전체 칼럼 그룹화
# 형식) DF.groupby(['칼럼1','칼럼2']) # 1차: 칼럼1 , 2차: 칼럼2
gender_smoker_grp = tips.groupby([tips['gender'],tips['smoker']])

# 4-1) 그룹객체.함수()
gender_smoker_grp.size() # 그룹별 빈도수
'''
gender  smoker
Female  No        54
        Yes       33
Male    No        97
        Yes       60 '''
gender_smoker_grp.mean() # 그룹별 평균
'''
               total_bill       tip      size   tip_pct
gender smoker                                          
Female No       18.105185  2.773519  2.592593  0.156921
       Yes      17.977879  2.931515  2.242424  0.182150
Male   No       19.791237  3.113402  2.711340  0.160669
       Yes      22.284500  3.051167  2.500000  0.152771 '''
gender_smoker_grp.describe() # 그룹별 요약통계량
'''
              total_bill                       ...   tip_pct                    
                   count       mean       std  ...       50%       75%       max
gender smoker                                  ...                              
Female No           54.0  18.105185  7.286455  ...  0.149691  0.181630  0.252672
       Yes          33.0  17.977879  9.189751  ...  0.173913  0.198216  0.416667
Male   No           97.0  19.791237  8.726566  ...  0.157604  0.186220  0.291990
       Yes          60.0  22.284500  9.911845  ...  0.141015  0.191697  0.710345 '''
gender_smoker_grp['tip'].describe() # 그룹별 특정변수(tip) 요약통계량
'''
               count      mean       std   min  25%   50%     75%   max
gender smoker                                                          
Female No       54.0  2.773519  1.128425  1.00  2.0  2.68  3.4375   5.2
       Yes      33.0  2.931515  1.219916  1.00  2.0  2.88  3.5000   6.5
Male   No       97.0  3.113402  1.489559  1.25  2.0  2.74  3.7100   9.0
       Yes      60.0  3.051167  1.500120  1.00  2.0  3.00  3.8200  10.0
[해설] : 여성은 흡연자, 남성은 비흡연자가 팁 지불에 후하다. '''


# 5. 집단변수 2개 -> 특정 칼럼 그룹화
gender_smoker_tip_grp = tips['tip'].groupby([tips['gender'],tips['smoker']])

# 5-1) 그룹별 빈도수 
gender_smoker_tip_grp.size() # 그룹별 빈도수
'''
gender  smoker
Female  No        54
        Yes       33
Male    No        97
        Yes       60 '''
gender_smoker_tip_grp.size().shape # 그룹별 빈도수 구조  # (4, ) : 1차원,series
grp_2d_size = gender_smoker_tip_grp.size().unstack() #  1d -> 2d 그룹별 tip 빈도수
grp_2d_size # 성별 vs 흡연유무  -> 교차분할표(빈도수)
'''
smoker  No  Yes
gender         
Female  54   33
Male    97   60 '''
grp_1d_size = grp_2d_size.stack() # 2d -> 1d 그룹별 tip 빈도수
grp_1d_size # 다시 원래대로
'''
gender  smoker
Female  No        54
        Yes       33
Male    No        97
        Yes       60 '''

# 5-2) 그룹별 합계
gender_smoker_tip_grp.sum() # 그룹별 tip 합계
'''
gender  smoker
Female  No        149.77
        Yes        96.74
Male    No        302.00
        Yes       183.07 '''
gender_smoker_tip_grp.sum().shape # (4, ) : 1차원,series
grp_2d = gender_smoker_tip_grp.sum().unstack() #  1d -> 2d 그룹별 tip 합계
grp_2d # 성별 vs 흡연유무  -> 교차분할표(합계)
'''
smoker      No     Yes
gender                
Female  149.77   96.74
Male    302.00  183.07 '''
gender_smoker_tip_grp.sum().unstack().shape # (2, 2) : 2차원,
grp_2d.shape # (2, 2)
grp_1d = grp_2d.stack() # 2d -> 1d 그룹별 tip 합계
grp_1d # 다시 원래대로
'''
gender  smoker
Female  No        149.77
        Yes        96.74
Male    No        302.00
        Yes       183.07 '''
grp_1d.shape # (4, )


# 6. iris dataset 그룹화
# 6-1) dataset load
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")

# 6-2) 그룹화
iris_grp = iris.groupby(iris['Species'])

# 6-3) 그룹별 함수 : group -> apply(sum)
iris_grp.sum() # 그룹별 합계(4개 변수)
'''
            Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
Species                                                         
setosa             250.3        171.4          73.1         12.3
versicolor         296.8        138.5         213.0         66.3
virginica          329.4        148.7         277.6        101.3 '''
iris_grp['Sepal.Width'].sum() # 그룹별 특정변수(sepal.width) 합계(1개 변수)
'''
Species
setosa        171.4
versicolor    138.5
virginica     148.7 '''
iris['Sepal.Width'].groupby(iris['Species']).sum() # 위와 같음











