import pandas as pd

pd.set_option('display.max_columns',None)                                                                   #모든 열 표시

#pd.set_option('display.max_rows',None)                                                                     #모든 행 표시
pd.set_option('display.max_seq_items', None)                                                                                                                    

mlb_pitchingstats = pd.read_csv(filepath_or_buffer="C:/Users/mink9/OneDrive/바탕 화면/졸업프로젝트/PITCHERS_TOTAL_MLB.csv",
                        encoding="utf_8",sep=",")

print(mlb_pitchingstats)


mlb_battingstats = pd.read_csv(filepath_or_buffer="C:/Users/mink9/OneDrive/바탕 화면/졸업프로젝트/BATTERS_TOTAL_MLB.csv",
                        encoding="utf_8",sep=",")

print(mlb_battingstats)
