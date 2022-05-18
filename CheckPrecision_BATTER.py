# =============================================================================
# SLG 예측
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns',None)                                                                   #모든 열 표시

#pd.set_option('display.max_rows',None)                                                                     #모든 행 표시
pd.set_option('display.max_seq_items', None)               

# TBF 상대 타자수
# FIP 수비무관 투수자책
# CG COMPLETE GAME
# SHO 완봉
# LOB% 잔루처리율
# BABIP 인플레이 타구 중 안타비율
# HBP 몸에 맞는 볼
# WP 폭투                                                                                                     

mlb_battingstats = pd.read_csv(filepath_or_buffer="./DATA/BATTERS_TOTAL_MLB.csv",
                        encoding="utf_8",sep=",")
#'Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K_9','BB_9','K_BB','H_9','HR_9','WHIP','BABIP','LOB_PCT','FIP'
batting_info = mlb_battingstats[['Age','G','AB','PA','H','R','RBI','IBB','SO','HBP','SF','AVG','OBP','SLG','BABIP']]

b_x = batting_info[['Age','G','AB','PA','H','R','RBI','IBB','SO','HBP','SF','AVG','OBP','BABIP']]
b_y = batting_info[['SLG']]

b_x_train, b_x_test, b_y_train, b_y_test = train_test_split(b_x, b_y, train_size=0.7,test_size=0.3)

b_model = LinearRegression()
b_model.fit(b_x_train, b_y_train)

b_y_predict = b_model.predict(b_x_test)


plt.scatter(b_y_test, b_y_predict, alpha=0.4)                                                               #그래프로 표현
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

print(b_model.score(b_x_train,b_y_train))


#여기서 부터는 선수 한명을 가지고 예측해보는 것임

kbo_battingstats = pd.read_csv(filepath_or_buffer="./DATA/BATTERS_TOTAL_KBO.csv",encoding="utf_8",sep=",")

#18~42
# =============================================================================
# age = 21
# 
# total_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','BB_9','H_9','HR_9','WHIP','BABIP','LOB_PCT','FIP']]
# total_ex = total_ex[total_ex['Age'] == age]
# print(total_ex)
# 
# b_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','BB_9','H_9','HR_9','WHIP','BABIP','LOB_PCT','FIP']]
# b_ex2 = b_ex[b_ex['Age'] == age]
# output_b = b_model.predict(b_ex2)
# print(output_b)
# print(min(output_b), max(output_b))
# =============================================================================

# =============================================================================
# age = 19
# for i in range(24):    
#     total_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','HR','BB','IBB','HBP','WP','BK','SO','BABIP','LOB_PCT','FIP']]
#     total_ex = total_ex[total_ex['Age'] == age]
# 
#     b_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K_9','BABIP','LOB_PCT','FIP']]
#     b_ex2 = b_ex[b_ex['Age'] == age]
# 
#     real_data = total_ex[['WHIP']]
# 
#     output_b = b_model.predict(b_ex2)
#     
#     arr = real_data - output_b
#     average = np.mean(arr)
# 
#     print(age,'세 :',average)
#     age = age + 1
#     
# ip = 10
# for i in range(10):
#     total_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K_9','WHIP','BABIP','LOB_PCT','FIP']]
#     total_ex = total_ex[(total_ex['IP'] >= ip+i*20) & (total_ex['IP'] < ip+(i+1)*20)]
# 
#     b_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K_9','BABIP','LOB_PCT','FIP']]
#     b_ex2 = b_ex[(b_ex['IP'] >= ip+i*20) & (b_ex['IP'] < ip+(i+1)*20)]
# 
#     real_data = total_ex[['WHIP']]
# 
#     output_b = b_model.predict(b_ex2)
#     
#     arr = real_data - output_b
#     average = np.mean(arr)
# 
#     print(ip+i*20,'~',ip+(i+1)*20,'이닝소화 :',average)
# =============================================================================


# 평균자책점 ERA
# 'Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','SO','WHIP','BABIP','LOB_PCT','FIP'




# =============================================================================
# 
# 
# total_ex = kbo_pitchingstats[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','SO','WHIP','BABIP','LOB_PCT','FIP']]
# total_ex = total_ex[total_ex['Age'] == 22]
# 
# b_ex = kbo_pitchingstats[['Age','W','L','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','SO','WHIP','BABIP','LOB_PCT','FIP']]
# b_ex2 = b_ex[(b_ex['Age'] == 22)]
# 
# #real_data = kbo_pitchingstats[['ERA']]
# real_data = total_ex[['ERA']]
# 
# output_b = b_model.predict(b_ex2)
# # =============================================================================
# # print(real_data)
# # print(output_b)
# # =============================================================================
# arr = real_data - output_b
# average = np.mean(arr)
# 
# print(average)
# =============================================================================

# =============================================================================
# 
# b_ex = kbo_pitchingstats[['Name','Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K_9','WHIP','BABIP','LOB_PCT','FIP']]
# #print(b_ex[(b_ex['Name'] == '박치국') & (b_ex['Age'] == 28)])
# print(b_ex[b_ex['Name'] == '이재학'])
# print('\n\n')
# #print(b_ex[(b_ex['Name'] == '양현종') & (b_ex['Age'] == 28)])
# 
# #b_ex2 = b_ex[(b_ex.Name == '박치국') & (b_ex.Age == 28)]
# b_ex2 = b_ex[b_ex.Name == '이재학']
# #b_ex2['Age'] = b_ex2['Age'].replace([27],[28])
# # =============================================================================
# # b_ex2['W'] = b_ex2['W'].replace([15],[10])
# # b_ex2['L'] = b_ex2['L'].replace([6],[12])
# # =============================================================================
# 
# input_b = b_ex2[['Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K_9','BABIP','LOB_PCT','FIP']]
# output_b = b_model.predict(input_b)
# print(output_b)  
# 
# =============================================================================



