import cv2
import numpy as np
import time

def trackbar(leftVideo,rightVideo):
    
    # open video
    video1 = cv2.VideoCapture(leftVideo)
    video2 = cv2.VideoCapture(rightVideo)
    
    if not video1.isOpened() or not video2.isOpened():
        print("Error: Cannot open video files")
        return
    
    # get total number of frames
    nr_of_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    nr_of_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    # create display window
    cv2.namedWindow("Video1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Video2", cv2.WINDOW_NORMAL)
    print("Controls:")
    print("  SPACE: Play/Pause")
    print("  'q' or ESC: Quit")
    print("  '+'/'-': Increase/Decrease speed")
    # set wait for each frame, determines playbackspeed
    playSpeed = 50
    paused = False
    
    def getFrame1(frame_nr):
        video1.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    def getFrame2(frame_nr):
        video2.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)    
    
    # function called by trackbar, sets the speed of playback
    def setSpeed(val):
        nonlocal playSpeed
        playSpeed = max(val, 1)
    
    # add trackbar
    cv2.createTrackbar("Frame1", "Video1", 0, nr_of_frames1, getFrame1)
    cv2.createTrackbar("Frame2", "Video2", 0, nr_of_frames2, getFrame2)
    
    # Read first frame to verify video is valid
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    
    if not ret1 or not ret2 or frame1 is None or frame2 is None:
        print("Error: Unable to read video frames")
        video1.release()
        video2.release()
        cv2.destroyAllWindows()
        return
    
    # Reset to beginning
    video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    video2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # main loop
    while 1:
        if not paused:
            # Get the next videoframe
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()
            
            if not ret1 or not ret2 or frame1 is None or frame2 is None:
                break
            
            h1, w1, c1 = frame1.shape
            h2, w2, c2 = frame2.shape
      
            if h1 > 500 or w1 > 500:
                frame1 = cv2.resize(frame1, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            
            if h2 > 500 or w2 > 500:
                frame2 = cv2.resize(frame2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
  
            # show frame
            if ret1:
                # Add pause indicator
                if paused:
                    cv2.putText(frame1, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Video1", frame1)
                # update slider position on trackbar
                try:
                    cv2.setTrackbarPos("Frame1", "Video1", int(video1.get(cv2.CAP_PROP_POS_FRAMES)))
                except:
                    pass

            if ret2:
                # Add pause indicator
                if paused:
                    cv2.putText(frame2, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Video2", frame2)
                # update slider position on trackbar
                try:
                    cv2.setTrackbarPos("Frame2", "Video2", int(video2.get(cv2.CAP_PROP_POS_FRAMES)))
                except:
                    pass
        else:
            # When paused, just refresh the display
            cv2.waitKey(10)   
        
        # display frame for 'playSpeed' ms, detect key input
        key = cv2.waitKey(playSpeed if not paused else 100)
        
        if key == ord('q') or key == 27:  # 'q' or ESC key
            print("Closing video windows...")
            break
        elif key == ord(' '):  # SPACE key for play/pause
            paused = not paused
            status = "PAUSED" if paused else "PLAYING"
            print(f"Status: {status}")
        elif key == ord('+') or key == ord('='):  # Speed up
            playSpeed = max(1, playSpeed - 10)
            print(f"Speed increased (delay: {playSpeed}ms)")
        elif key == ord('-') or key == ord('_'):  # Speed down
            playSpeed = min(200, playSpeed + 10)
            print(f"Speed decreased (delay: {playSpeed}ms)")
        
        # Check if windows are closed by user
        try:
            if cv2.getWindowProperty("Video1", cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty("Video2", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user")
                break
        except:
            break
    
    # release resources
    print("Releasing resources...")
    video1.release()
    video2.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Process any pending GUI events

 