import re
import os
import numpy as np
path = "C:/Users/user/Desktop/video/stitching_tracking/"
paths = [os.path.join(path , i ) for i in os.listdir(path)]
store1 = []
store2 = []
for i in paths :
    if len(i) == 19 :
        store2.append(i)
    else :
        store1.append(i)

paths = list(np.sort(store1)) + list(np.sort(store2))
print(paths)
pathIn= 'C:/Users/user/Desktop/video/stitching_tracking/'
pathOut = 'C:/Users/user/Desktop/video/stitching_tracking/result.mp4'
fps = 24
import cv2
frame_array = []
for idx , path in enumerate(paths) : 
    if (idx % 2 == 0) | (idx % 5 == 0) :
        continue
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()