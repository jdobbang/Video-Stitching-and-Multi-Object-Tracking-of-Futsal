import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import pandas as pd
import seaborn as sns
import math
import cv2

def mov(player):

  person = pd.read_csv('person_upgraded_cylindrical.csv')
  ball = pd.read_csv('sports_ball_upgraded_cylindrical.csv')

  data = []
  for index in range(4):
    a = person[person['id']==index+1]
    a = a.reset_index()
    total_dist = 0
    for i in a.index:
      if i == a.shape[0]-1:
        break
      dist = math.sqrt((a['x'][i+1] - a['x'][i])**2 + (a['y'][i+1] - a['y'][i])**2)
      total_dist +=dist
    element = 'ID'+ str(index+1) + ' 활동량 : ' + str(round(total_dist)) + 'm' + '       평균속도  : '+ str(round(total_dist/30)) + 'm/s'
    data.append(element)
    #print('ID',index+1,'활동량',total_dist,'m','/ 평균속도',total_dist/30,'m/s','\n')
  print('id',player)
  return data[player-1]
  #print(data)