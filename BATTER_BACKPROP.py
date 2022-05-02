import pandas as pd
import pybaseball as pb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#아래의 코드는 지표들을 활용하여 AVG(안타율)만을 예측하는 것이다. 나머지 지표 예측은 반영할 지표들을 수정하여 비슷한 방법으로 향후 할 것

pd.set_option('display.max_columns',None)                                                                   #모든 열 표시

#pd.set_option('display.max_rows',None)                                                                     #모든 행 표시
pd.set_option('display.max_seq_items', None)                                                                #생략없이 야구 선수 columns표현                                                                                                                      

batting_info = pb.batting_stats(start_season=2020,end_season=(2021))[['Age','AVG','OBP','SLG','G','PA','AB','R','H','1B','2B','3B','HR','RBI','SB','CS','BB','HBP','IBB','SO','GDP','SH','SF']]
                                                                                                          #사용할 야구선수를 설정하고 사용할 데이터도 설정, 1955부터 SF, IBB, *GDP 매기기 시

#print(batting_info.count())

b_x = batting_info[['Age','OBP','SLG','G','PA','AB','R','H','1B','2B','3B','HR','RBI','SB','CS','BB','HBP','IBB','SO','GDP','SH','SF']]
b_y = batting_info[['AVG']]

b_x_train, b_x_test, b_y_train, b_y_test = train_test_split(b_x, b_y, train_size=0.8,test_size=0.2)

b_model = LinearRegression()                                                                                  #다중선형회귀 모델 생성
b_model.fit(b_x_train, b_y_train)

b_y_predict = b_model.predict(b_x_test)


plt.scatter(b_y_test, b_y_predict, alpha=0.4)                                                               #그래프로 표현
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
                                                                                                          
print(b_model.score(b_x_test,b_y_test))                                                                     #이 모델의 정확도

b_weights = b_model.coef_                                                                                   #이 모델의 가중치 값들


b_csv = pd.read_csv("./BATTERS_TOTAL_KBO.csv")                                                              #KBO 타자 데이터
b_kbo_x = b_csv[['Age','OBP','SLG','G','PA','AB','R','H','1B','2B','3B','HR','RBI','SB','CS','BB','HBP','IBB','SO','GDP','SH','SF']]
b_kbo_y = b_csv[['AVG']]

b_kbo_predict = b_model.predict(b_kbo_x)
plt.scatter(b_kbo_y, b_kbo_predict, alpha=0.4)                                                               #그래프로 표현
plt.xlabel("Real")
plt.ylabel("Predicted KBO")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

print(b_model.score(b_kbo_x,b_kbo_y))                                                                        #현재 MLB모델로 예측한 KBO 선수의 성적 예측 정확도


kbo_data = b_csv                                                                                             #행렬을 사용해서 계산할 것이므로 torch 이용 
x_data = torch.FloatTensor(np.array(b_kbo_x))
y_data = torch.FloatTensor(np.array(b_kbo_y))                                      

epochs = 30000
                                                                                                           # 가중치값은 MLB에서 만든 것을 사
targets_data = b_weights.T
targets_df = pd.DataFrame(data=targets_data)
targets_df.columns = ['targets']
W = torch.Tensor(np.array(b_weights).T)
W = W.requires_grad_(True)
b = torch.zeros(1, requires_grad= True)

optimizer = optim.SGD([W,b],lr=1e-6)                                                                        #lr = learnig rate인데 값 적절하게 작게 해야됨 너무 크면 오류남 


for epoch in range(epochs + 1):
    hypothesis = x_data.matmul(W) + b
    
    cost = torch.mean((hypothesis - y_data)**2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} '.format(
        epoch, epochs, hypothesis.squeeze().detach(), cost.item()
    ))
    if epoch == epochs:
        hypo = hypothesis.detach().numpy()

hypo = np.ravel(hypo,order='C')
total=0
for s in range(len(hypo)):
    total += hypo[s]

b_kbo_y = np.array(b_kbo_y)
b_kbo_y = np.ravel(b_kbo_y,order='C')
total2 = 0
for p in range(len(b_kbo_y)):
    total2 += b_kbo_y[p]

kbo_score = 1 - ( abs(total - total2) / (total2) )

print("MLB모델로 MLB선수들 평가 할떄의 정확도 =                {:.6f}".format(b_model.score(b_x_test,b_y_test)))
print("MLB모델을 튜닝하지 않고 KBO선수들 평가 할떄의 정확도 =   {:.6f}".format(b_model.score(b_kbo_x,b_kbo_y)))
print('MLB모델을 튜닝하여 KBO선수들 평가 할떄의 정확도 =        {:.6f}'.format(np.mean(kbo_score)))
print('계산된 최종 가중치 = ')
print(W)











