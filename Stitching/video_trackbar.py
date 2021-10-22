import cv2
import numpy as np
import time

def trackbar(leftVideo,rightVideo):
    
    # open video
    video1 = cv2.VideoCapture(leftVideo)
    video2 = cv2.VideoCapture(rightVideo)
    # get total number of frames
    nr_of_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    nr_of_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    # create display window
    cv2.namedWindow("Video1")
    cv2.namedWindow("Video2")
    # set wait for each frame, determines playbackspeed
    playSpeed = 50
    
    def getFrame1(frame_nr):
        #global video
        video1.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    def getFrame2(frame_nr):
        #global video
        video2.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)    
#  function called by trackbar, sets the speed of playback
    def setSpeed(val):
        #global playSpeed
        playSpeed = max(val,1)
    
    # add trackbar
    cv2.createTrackbar("Frame1", "Video1", 0,nr_of_frames1,getFrame1)
    cv2.createTrackbar("Frame2", "Video2", 0,nr_of_frames2,getFrame2)
    # main loop
    while 1:
        # Get the next videoframe
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        
        h1, w1, c1 = frame1.shape
        h2, w2, c2 = frame2.shape
      
        if h1 > 500 or w1 > 500:
            frame1 = cv2.resize(frame1, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) #일단 임의로 사이즈 조절
        
        if h2 > 500 or w2 > 500:
            frame2 = cv2.resize(frame2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) #일단 임의로 사이즈 조절
  
        # show frame, break the loop if no frame is found
        if ret1:
            cv2.imshow("Video1", frame1)
            # update slider position on trackbar
            # NOTE: this is an expensive operation, remove to greatly increase max playback speed
            cv2.setTrackbarPos("Frame1","Video1", int(video1.get(cv2.CAP_PROP_POS_FRAMES)))
        else:
            break

        if ret2:
            cv2.imshow("Video2", frame2)
            # update slider position on trackbar
            # NOTE: this is an expensive operation, remove to greatly increase max playback speed
            cv2.setTrackbarPos("Frame2","Video2", int(video2.get(cv2.CAP_PROP_POS_FRAMES)))
        else:
            break   
        
        # display frame for 'playSpeed' ms, detect key input
        key = cv2.waitKey(playSpeed)
        
        if key == ord('q'):
            break
        if key == ord('p'):
            time.sleep(5)
        
    # release resources
    video1.release()
    video2.release()
    cv2.destroyAllWindows()

 