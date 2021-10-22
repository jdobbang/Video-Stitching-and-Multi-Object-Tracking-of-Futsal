import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from matplotlib.patches import Arc
import pandas as pd
import seaborn as sns
import math
import cv2
import glob
fig=plt.figure()

def createPitch():
    
    #Create figure
    
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


def mini():
    person = pd.read_csv('person_upgraded_cylindrical.csv')
    ball = pd.read_csv('sports_ball_upgraded_cylindrical.csv')

    for index in range(person['frame'].max()):
        createPitch()
        a= person[person['frame']==index+1]
        
        xb = ball[ball['frame']==index+1].x*28
        yb = ball[ball['frame']==index+1].y*28
        
        x1 = a[a['id']==1].x*28
        y1 = a[a['id']==1].y*28

        x2 = a[a['id']==2].x*28
        y2 = a[a['id']==2].y*28

        x3 = a[a['id']==3].x*28
        y3 = a[a['id']==3].y*28
        
        x4 = a[a['id']==4].x*28
        y4 = a[a['id']==4].y*28
        
        plt.scatter(xb,yb, cmap ='Oranges')
        plt.scatter(x1,y1, cmap = 'Blues')
        plt.scatter(x2,y2, cmap = 'Blues')
        plt.scatter(x3,y3, cmap = 'Blues')
        plt.scatter(x4,y4, cmap = 'Blues')
        
        #plt.text(xb,yb, 'Ball')
        #plt.text(x1,y1, '1')
        #plt.text(x2,y2, '2')
        #plt.text(x3,y3, '3')
        #plt.text(x4,y4, '4')

        plt.xlim(0,454)
        plt.ylim(0,200)

        filename = str('./static/minimap/frame') + str(index+1).zfill(5)
        plt.savefig(filename)

def f2v():
    img_array = []
    for filename in glob.glob('./static/minimap/*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter('./static/video/minimap.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
if __name__ == '__main__':
    mini()
    f2v()