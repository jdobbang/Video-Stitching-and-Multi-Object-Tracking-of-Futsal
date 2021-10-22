#flask importing
from io import StringIO
from flask import Flask, render_template, send_file, make_response, url_for, Response
from flask import request
from flask import make_response
#Pandas and Matplotlib
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#other requirements
import io
import seaborn as sns
from matplotlib.patches import Arc
import draw
import movement as mv

#data imports
person = pd.read_csv("./person_upgraded_cylindrical.csv")
ball = pd.read_csv("./sports_ball_upgraded_cylindrical.csv")

#initializing the app
app = Flask(__name__,template_folder = 'templates',static_folder = 'static')

@app.route('/', methods=['GET','POST'])

def index():

    if request.method == 'GET':
        return (render_template('test_html.html'))
    
    if request.method == 'POST':
        
        userID = request.form["userID"]
        gamePW = int(request.form["gamePW"])
        playerID = int(request.form["playerID"])
        
        #팀 하려면 option 걸기
        
        #임시 조회 정보 입력
        if userID == 'supput' and gamePW == 0000:
          state = '조회 성공'
          draw.heatmap(playerID)
          draw.average(playerID)
          data = mv.mov(playerID)
      
        else:
          state = '조회 실패'
           
    return render_template('result_html.html',id = playerID, result = state, movement = data, url = '/static/images/new_plot.png', url2 = '/static/images/new_plot2.png')   
      
if __name__ == '__main__':
    app.run(host='0.0.0.0',port = '8000')
    # 도현 desktop에서 실행됨
    # 외부 ip 주소 58.141.183.46
            