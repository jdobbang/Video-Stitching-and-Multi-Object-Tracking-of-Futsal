
# Import Libraries
import numpy as np
import cv2

# Get the source video file
videoFileName = "./output/CylindricalOutput.avi"
cap = cv2.VideoCapture(videoFileName)

# Set the width and height parameters for output video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter('Track-the-ball.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, (width,height))

# Color code for the  blue rectangle
blue = (255,128,0)

# Create the tracker
tracker = cv2.legacy.TrackerCSRT_create()

# Initializers
initi = 0
ok = False
goprocess = 0
r = 0

print("Processing the input video file:")

# When the cap is  opened
while(cap.isOpened()):
    
    ret, frame = cap.read()   
    print("while문!") 
    # If read return value is true process further     
    
    if ret == True:        
        
        # Read image as gray-scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur the image to reduce noise, kernel window is seletced as 5
        img_blur = cv2.medianBlur(gray, 5)
        
        # Apply hough transform on the gray scale image
        # Change the parameters to match the circle that is needed
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 450, param1=300, param2=20, minRadius=20, maxRadius=30)
        
        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:             
                # Change the parameters to select the circle that is needed in the frame
                # Here I am selecting only the circles above 300 px in the x axis   
                if i[0] > 300:
                    #print('Circle center - {} , {} -- Radius {}'.format(i[0],i[1],i[2]))
                    # Draw circle around the ball
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)                    
                    r = i[2]
                    # Readjust the parameter to initilize the track on the right circle.
                    if initi == 0:
                        # identidy the bounding box's points from the circle's point and radius
                        bbox = (i[0]+i[2], i[1]-i[2], 2*i[2], 2*i[2])   
                        ok = tracker.init(frame, bbox)
                        initi = 1
                        goprocess = 1
        
        # Allow the tracker to process once it is initialized
        if goprocess == 1:
            ok, bbox = tracker.update(frame)
            #print("{} , {} ".format(round(int(bbox[3])/2), r)) 
            
            # Try to readjust the bound box side to that of the circle's radius
            # Redetection and tracking , happens here
            if ok and round(int(bbox[3])/2) == r:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, blue, 2, 2 )    
            else:
                # reinitialize the tracker based on detection
                tracker.clear()
                #bbox = (x, y, w, h)                 
                ok = tracker.init(frame, bbox)
                ok, bbox = tracker.update(frame)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, blue, 2, 2 )  
                #print("{} , {} ".format(round(int(bbox[3])/2), r)) 
                                                  
        #Write the frame to the output video file
        #out.write(frame)
        cv2.imshow("frame",frame)
        cv2.waitKey(30) 
    else:
        print("Processing is completed")
        break
            