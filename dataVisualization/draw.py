import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import pandas as pd
import seaborn as sns
import math
import cv2

def createPitch():
    
    #Create figure
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,200], color="black")
    plt.plot([0,454],[200,200], color="black")
    plt.plot([454,454],[200,0], color="black")
    plt.plot([454,0],[0,0], color="black")
    plt.plot([227,227],[0,200], color="black")

     #Prepare Circles
    centreCircle = plt.Circle((227,100),20,color="black",fill=False)
    centreSpot = plt.Circle((227,100),2,color="black")
    
    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)

person = pd.read_csv('person_upgraded_cylindrical.csv')
#ball = pd.read_csv('sports_ball_upgraded_cylindrical.csv')

def heatmap(id):
     
    createPitch()
    #plt.show()
    #Create the heatmap
    kde = sns.kdeplot(
            data=person[person['id']==id],
            x="x",
            y="y",
            shade_lowest=False,
            shade = True
    )
    
    plt.xlim(0,16.5)
    plt.ylim(0,7.5)
    #plt.show()
    
    plt.savefig('./static/images/new_plot.png')

def average(id):
    createPitch()

    average_x = person[person['id']==id].x.mean()*28
    average_y = person[person['id']==id].y.mean()*28

    plt.scatter(average_x, average_y, s=200)
    plt.title('average location')

    plt.xlim(0,454)
    plt.ylim(0,200)
    
    plt.savefig('./static/images/new_plot2.png')

if __name__ == '__main__':
    create()

