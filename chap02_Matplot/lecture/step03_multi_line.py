# -*- coding: utf-8 -*-
"""
step03_multi_line
 - marker, color, line style, label
"""
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot') # 차트 격자 제공 


# 1. 

# 1-1) data 생성
data1 = 0.2 + 0.1 * np.random.randn(100)
data2 = 0.4 + 0.2 * np.random.randn(100)
data3 = 0.6 + 0.3 * np.random.randn(100)
data4 = 0.8 + 0.4 * np.random.randn(100)

# 1-2) 격자생성
fig = plt.figure(figsize=(12,8))
chart = fig.add_subplot()

# 1-3) 선형차트 생성
chart.plot(data1, marker='o', color='black', linestyle='-.', label='data1')
chart.plot(data2, marker='^', color='red', linestyle='-', label='data2')
chart.plot(data3, marker='*', color='green', linestyle='--', label='data3')
chart.plot(data4, marker='+', color='blue', linestyle=':', label='data4')
plt.legend(loc='best') # 범례
chart.set_title('Multi-line chart')
chart.set_xlabel('색인')
chart.set_ylabel('random number')
plt.show()



















































