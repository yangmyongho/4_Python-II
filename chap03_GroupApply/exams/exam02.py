'''
문2) weatherAUS.csv 데이터셋을 이용하여 다음 단계를 해결하시오.
   <단계1> 지역(Location)별 빈도수 구하기
     사용 함수 : value_counts()
     
   <단계2> 2개 칼럼(Location과  RainToday) -> DF 전체 칼럼 그룹화
     사용 함수 : groupby()
   <단계3> 그룹화 객체를 대상으로 평균 구하기   
                                  MinTemp    MaxTemp  ...    Temp3pm    RISK_MM
    Location      RainToday                        ...                      
    Adelaide      No         13.340847  24.313011  ...  22.917576   0.927778
                  Yes        11.426263  17.780402  ...  16.484422   3.426667
    Albany        No         13.433688  21.001590  ...  19.294128   1.488339
                  Yes        11.560593  17.345763  ...  15.878723   4.282627
                  
   <단계4> 그룹화 결과를 대상으로 테이블 형식으로 변경 
     사용 함수 : size().unstack()
    <<출력 결과>>
     RainToday          No  Yes
     Location                  
     Adelaide          661  199
     Albany            566  236
     Albury            615  169
     AliceSprings      693   99
     BadgerysCreek     596  158
     Ballarat          586  223
     Bendigo           658  153 
          : 
   <단계5> 단계4의 결과를 대상으로 가로막대 누적형 차트 그리기
'''
import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv("C:/ITWILL/4_Python-II/data/weatherAUS.csv")
weather
weather.info()

# 단계 1 
#weather['Location'].unique() # 종류는나오지만 갯수는 나오지않음 
weather.Location.value_counts()
'''
Canberra            1085
Sydney               971
Melbourne            904
Perth                899
Hobart               893
Brisbane             887
Adelaide             887
Darwin               881
Ballarat             813
Bendigo              811
Cairns               805
Albany               802
SydneyAirport        801
NorahHead            796
Albury               795
Launceston           795
Williamtown          794
Moree                793
MountGambier         793
AliceSprings         792
MountGinini          789
BadgerysCreek        788
Tuggeranong          782
Newcastle            782
Townsville           777
GoldCoast            777
MelbourneAirport     774
Mildura              769
Richmond             769
Walpole              768
Woomera              766
Witchcliffe          765
CoffsHarbour         765
Portland             763
Cobar                763
PearceRAAF           763
SalmonGums           759
Dartmoor             757
NorfolkIsland        756
Nuriootpa            756
WaggaWagga           755
Watsonia             755
Wollongong           752
Penrith              750
Sale                 746
PerthAirport         738
Name: Location, dtype: int64 '''

# 단계 2
weather_grp = weather.groupby(['Location','RainToday'])
weather_grp

# 단계 3
grp_mean = weather_grp.mean()
grp_mean
'''
                         MinTemp    MaxTemp  ...    Temp3pm   RISK_MM
Location    RainToday                        ...                     
Adelaide    No         13.340847  24.313011  ...  22.917576  0.927778
            Yes        11.426263  17.780402  ...  16.484422  3.426667
Albany      No         13.433688  21.001590  ...  19.294128  1.488339
            Yes        11.560593  17.345763  ...  15.878723  4.282627
Albury      No          9.859283  24.100493  ...  22.981789  1.263816
                         ...        ...  ...        ...       ...
Witchcliffe Yes        10.238614  17.568317  ...  15.618812  5.864356
Wollongong  No         14.960440  21.688419  ...  20.099069  1.997363
            Yes        14.817778  20.053889  ...  18.818539  8.737079
Woomera     No         13.565385  26.901994  ...  25.491738  0.313960
            Yes        12.393220  22.220339  ...  20.548276  1.932203 '''

# 단계 4 
grp_size_tab = weather_grp.size().unstack()        
grp_size_tab
'''
RainToday          No  Yes
Location                  
Adelaide          661  199
Albany            566  236
Albury            615  169
AliceSprings      693   99
BadgerysCreek     596  158
Ballarat          586  223
Bendigo           658  153
Brisbane          665  218
Cairns            543  262
Canberra          877  202
Cobar             652  107
CoffsHarbour      520  244
Dartmoor          463  247
Darwin            616  265
GoldCoast         554  222
Hobart            665  227
Launceston        601  194
Melbourne         656  203
MelbourneAirport  603  171
Mildura           677   92
Moree             668  123
MountGambier      564  229
MountGinini       497  222
Newcastle         575  197
NorahHead         561  205
NorfolkIsland     536  220
Nuriootpa         598  158
PearceRAAF        560  128
Penrith           572  139
Perth             721  178
PerthAirport      606  132
Portland          486  273
Richmond          601  149
Sale              591  155
SalmonGums        622  115
Sydney            722  248
SydneyAirport     584  217
Townsville        616  160
Tuggeranong       611  157
WaggaWagga        616  139
Walpole           500  229
Watsonia          564  190
Williamtown       522  202
Witchcliffe       546  202
Wollongong        547  182
Woomera           703   59 '''

# 단계 5
grp_size_tab.plot(kind='barh', title='LOCATION vs RAINTODAY', stacked=True)
plt.show()






















