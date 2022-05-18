import pandas as pd
import pybaseball as pb

pd.set_option('display.max_columns',None)                                                                   #모든 열 표시
                                                                   #모든 행 표시
pd.set_option('display.max_seq_items', None)                                                                #생략없이 야구 선수 columns표현                                                                                                                      

pitching_info = pb.pitching_stats(start_season=2002,end_season=(2021),qual=10)[['Name','Age','W','L','ERA','G','GS','CG','ShO','SV','BS','HLD','IP','TBF','H','R','ER','HR','BB','IBB','HBP','WP','BK','SO','K/9','BB/9','K/BB','H/9','HR/9','WHIP','BABIP','LOB%','FIP']]



pitching_info.to_csv("C:/Users/workstation/OneDrive/바탕 화면/정민규/pitcher_stat_advanced/PITCHERS_TOTAL_MLB.csv", mode = 'w')

# =============================================================================
# batting_info = pb.batting_stats(start_season=2002,end_season=(2021),qual=30)[['Name','Age','G','AB','PA','H','HR','1B','2B','3B','HR','R','RBI','BB','IBB','SO','HBP','SF','SH','GDP','SB','CS','AVG','OBP','SLG','OPS','ISO','BABIP']]
# 
# #'Age','G','AB','PA','H','HR','1B','2B','3B','HR','R','RBI','BB','IBB','SO','HBP','SF','SH','GDP','SB','CS','AVG','OBP','SLG','OPS','ISO','BABIP'
# 
# batting_info.to_csv("C:/Users/workstation/OneDrive/바탕 화면/정민규/batter_stat_advanced/BATTERS_TOTAL_MLB.csv", mode = 'w')
# =============================================================================
